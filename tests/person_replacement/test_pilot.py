import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from r2v_data_v2.person_replacement import bernini, pipeline, qwen3vl
from tools.person_replacement.run_bernini_pilot import main


def inputs(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    video = clips / "episode_0.mp4"
    video.write_bytes(b"mock video")
    source = tmp_path / "collected_face_1.jsonl"
    source.write_text(json.dumps({
        "video_path": video.name, "source_video_path": "/never/use/full.mov",
        "source_video_id": "episode",
    }) + "\n")
    return source, clips, video


def test_prompt_and_official_case_are_plain_strings(tmp_path):
    prompt = pipeline.build_prompt("a gray-clothed person", "a green-clothed person")
    assert prompt.startswith("Replace a gray-clothed person with a green-clothed person.")
    assert "mouth state" in prompt and "non-human regions unchanged" in prompt
    assert bernini.serialize_case(tmp_path / "real.mp4", tmp_path / "replacement.mp4", prompt) == {
        "task_type": "v2v", "video": str(tmp_path / "real.mp4"),
        "output": str(tmp_path / "replacement.mp4"), "prompt": prompt,
    }


def test_real_resolver_uses_explicit_root_not_source_video(tmp_path):
    source, clips, video = inputs(tmp_path)
    selected = pipeline.select_cases(source, clips, tmp_path / "output", limit=1)
    assert selected[0]["target_video_path"] == str(video)
    source.write_text('{"video_path":"../outside.mp4"}\n')
    with pytest.raises(ValueError, match="escapes"):
        pipeline.select_cases(source, clips, tmp_path / "output", limit=1)


def test_dry_run_with_mock_resolver_is_read_only_and_model_free(tmp_path, monkeypatch, capsys):
    source, clips, video = inputs(tmp_path)
    seen = []

    def resolve(self, raw):
        seen.append(raw["video_path"])
        return video, video.name

    monkeypatch.setattr(pipeline.JeaVideoMotionAdapter, "resolve_clip_path", resolve)
    monkeypatch.setattr(qwen3vl.LocalQwen, "_load", lambda self: pytest.fail("model loaded"))
    monkeypatch.setattr(bernini.BerniniAdapter, "run", lambda *a: pytest.fail("Bernini called"))
    output = tmp_path / "output"
    assert main(["--input-jsonl", str(source), "--clips-root", str(clips),
                 "--output-root", str(output), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert seen == [video.name]
    assert report["cases"][0]["target_video_path"] == str(video)
    assert not output.exists()


def test_frame_zero_uses_existing_decoder_and_exact_index(tmp_path, monkeypatch):
    calls = []

    class Decoder:
        def decode_indices(self, video, indices):
            calls.append((video, indices))
            return [SimpleNamespace(image=Image.new("RGB", (7, 9), "red"))]

    monkeypatch.setattr(pipeline, "OpenCvFrameDecoder", Decoder)
    video, output = tmp_path / "clip.mp4", tmp_path / "frame0.jpg"
    pipeline.extract_frame_zero(video, output)
    assert calls == [(video, [0])]
    with Image.open(output) as image:
        assert image.size == (7, 9)


def test_manifest_pairs_replacement_with_original_and_releases_qwen(tmp_path, monkeypatch):
    source, clips, video = inputs(tmp_path)
    cases = pipeline.select_cases(source, clips, tmp_path / "output", limit=1)
    events = []

    class Qwen:
        model_path = tmp_path / "qwen"

        def describe(self, path):
            assert path == video
            events.append("describe")
            return "source person"

        def invent(self, description):
            assert description == "source person"
            events.append("invent")
            return "different person"

        def close(self):
            events.append("release qwen")

    class Bernini:
        config = tmp_path / "bernini-model"

        def validate(self):
            pass

        def run(self, case_path, seed):
            events.append("bernini")
            assert seed == 37
            case = json.loads(case_path.read_text())
            assert case["video"] == str(video)
            Path(case["output"]).write_bytes(b"generated video")

    monkeypatch.setattr(pipeline, "extract_frame_zero", lambda v, p: p.write_bytes(b"jpeg"))
    monkeypatch.setattr(pipeline, "validate_video", lambda p: None)
    manifests = pipeline.run_cases(cases, qwen=Qwen(), bernini=Bernini(), seed=37)
    assert events == ["describe", "invent", "release qwen", "bernini"]
    manifest = json.loads(manifests[0].read_text())
    assert manifest["target_video_path"] == str(video)
    assert manifest["input_video_path"].endswith("replacement.mp4")
    assert manifest["reference_image_path"].endswith("reference_frame0.jpg")
    assert manifest["seed"] == 37
    assert video.read_bytes() == b"mock video"
    with pytest.raises(FileExistsError):
        pipeline.run_cases(cases, qwen=Qwen(), bernini=Bernini(), seed=37)


def test_qwen_import_and_constructor_do_not_load_models(tmp_path):
    script = """
import sys
from r2v_data_v2.person_replacement.qwen3vl import LocalQwen
LocalQwen('/does/not/exist')
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_qwen_two_calls_one_local_model_and_plain_text(tmp_path, monkeypatch):
    events = []

    class Batch(dict):
        def to(self, device):
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"]
            events.append("processor")
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            events.append((messages, kwargs))
            return Batch(input_ids=[[10, 11]])

        def batch_decode(self, ids, **kwargs):
            assert ids == [[12]]
            return [" concise description "]

    class Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert kwargs["local_files_only"] and kwargs["device_map"] == "auto"
            events.append("model")
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 128 and kwargs["do_sample"] is False
            return [[10, 11, 12]]

    from contextlib import nullcontext
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=Processor, Qwen3VLForConditionalGeneration=Model,
    ))
    video = tmp_path / "clip.mp4"
    video.touch()
    client = qwen3vl.LocalQwen(tmp_path)
    assert client.describe(video) == client.invent("source person") == "concise description"
    assert events.count("model") == events.count("processor") == 1
    requests = [e for e in events if isinstance(e, tuple)]
    assert requests[0][0][0]["content"][0] == {"type": "video", "path": str(video)}
    assert "num_frames" not in requests[0][1]
    assert "fps" not in requests[0][1]
    assert requests[1][0][0]["content"][0]["type"] == "text"
    assert "num_frames" not in requests[1][1]
    assert "fps" not in requests[1][1]


def test_launcher_demo_loop_is_one_case_and_cli_seed_without_modifying_source(tmp_path, monkeypatch):
    code = tmp_path / "official"
    launcher = code / "scripts/bernini/run_v2v.sh"
    launcher.parent.mkdir(parents=True)
    launcher.write_text('''#!/usr/bin/env bash
set -euo pipefail
CASE_PATH=${CASE_PATH:-assets/testcases/v2v/v2v_case3.json}
for CASE_PATH in assets/testcases/v2v/v2v_case1.json assets/testcases/v2v/v2v_case2.json assets/testcases/v2v/v2v_case3.json;
do
BERNINI_CONFIG=${BERNINI_CONFIG:-ByteDance/Bernini-Diffusers}
torchrun --standalone --nproc-per-node 8 infer_multi_gpu.py --config "$BERNINI_CONFIG" --seed 42 --case "$CASE_PATH"
done
''')
    (code / "infer_multi_gpu.py").touch()
    model = tmp_path / "checkpoint"
    model.mkdir()
    before = launcher.read_bytes()
    effective = bernini.prepare_launcher(launcher.read_text(), 19)
    assert 'for CASE_PATH' not in effective
    assert '--seed 19' in effective
    assert launcher.read_bytes() == before
    assert effective.count('torchrun') == 1
    with pytest.raises(ValueError):
        bernini.prepare_launcher(launcher.read_text().replace('--case', '--use_pe --case'), 19)
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    generated = case_dir / "replacement.mp4"
    case_path = case_dir / "bernini_case.json"
    case_path.write_text(json.dumps(bernini.serialize_case(tmp_path / "clip.mp4", generated, "edit")))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_torchrun = fake_bin / "torchrun"
    fake_torchrun.write_text(f'''#!{sys.executable}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
case_path = Path(args[args.index('--case') + 1])
case = json.loads(case_path.read_text())
with (case_path.parent / 'invocations.jsonl').open('a') as f:
    f.write(json.dumps({{'args': args, 'cwd': os.getcwd(), 'offline': os.environ['HF_HUB_OFFLINE']}}) + '\\n')
Path(case['output']).write_bytes(b'fake generated video')
''')
    fake_torchrun.chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    bernini.BerniniAdapter(code, model).run(case_path, 19)
    calls = (case_dir / "invocations.jsonl").read_text().splitlines()
    assert len(calls) == 1  # no three-demo loop
    call = json.loads(calls[0])
    assert call["args"][call["args"].index("--seed") + 1] == "19"
    assert call["args"][call["args"].index("--config") + 1] == str(model)
    assert call["cwd"] == str(code) and call["offline"] == "1"
    assert generated.read_bytes() == b"fake generated video"
    assert launcher.read_bytes() == before


def test_failed_generation_never_publishes_success_manifest(tmp_path, monkeypatch):
    source, clips, _ = inputs(tmp_path)
    cases = pipeline.select_cases(source, clips, tmp_path / "out", limit=1)
    qwen = SimpleNamespace(model_path=tmp_path, describe=lambda v: "source",
                           invent=lambda d: "replacement", close=lambda: None)
    adapter = SimpleNamespace(config=tmp_path, validate=lambda: None, run=lambda p, s: None)
    monkeypatch.setattr(pipeline, "extract_frame_zero", lambda v, p: p.write_bytes(b"jpeg"))
    with pytest.raises(ValueError, match="no video"):
        pipeline.run_cases(cases, qwen=qwen, bernini=adapter, seed=42)
    assert not (Path(cases[0]["output_dir"]) / "manifest.json").exists()


def test_launcher_refuses_unrecognized_case_override():
    script = 'CASE_PATH=wrong.json\nBERNINI_CONFIG=wrong-model\ntorchrun infer_multi_gpu.py --config "$BERNINI_CONFIG" --case "$CASE_PATH" --seed 42\n'
    with pytest.raises(ValueError, match="override"):
        bernini.prepare_launcher(script, 42)


def test_missing_bernini_root_has_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("BERNINI_CODE_ROOT", raising=False)
    with pytest.raises(ValueError, match="bernini-code-root"):
        bernini.BerniniAdapter(None, tmp_path).validate()


@pytest.mark.parametrize("path", ["/mnt/workspace/public/dataset/out", "/mnt/workspace/public/pretrained/out", "/mnt/workspace/liutao/X_human_data/out"])
def test_output_cannot_write_protected_inputs(path):
    with pytest.raises(ValueError, match="read-only"):
        pipeline.validate_output_root(Path(path))


def test_input_two_is_not_silently_processed(tmp_path):
    with pytest.raises(ValueError, match="collected_face_2"):
        pipeline.select_cases(tmp_path / "collected_face_2.jsonl", tmp_path, tmp_path / "out", limit=1)
