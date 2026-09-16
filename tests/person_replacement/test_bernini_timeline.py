import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.person_replacement import bernini, pipeline, timeline


def source(count=138, num=25, den=1):
    return timeline.VideoTimeline(count, num / den, num, den, 640, 480, count * den / num)


@pytest.mark.parametrize("count,fps,internal,pad", [(81,16,81,0), (137,25,137,0),
                                                    (138,25,141,3), (1,25,5,4)])
def test_plan_matches_confirmed_upstream_factor_minimum(count, fps, internal, pad):
    plan = bernini.plan_bernini_timeline(source(count, fps))
    assert (plan.internal_frame_count, plan.internal_fps, plan.pad_frames) == (internal, fps, pad)


@pytest.mark.parametrize("num,integer", [(24000,24), (30000,30)])
def test_fractional_plan_requires_integer_input_even_without_padding(num, integer):
    plan = bernini.plan_bernini_timeline(source(1201, num, 1001))
    assert plan.internal_frame_count == 1201 and plan.internal_fps == integer
    assert plan.pad_frames == 0 and plan.requires_input_conversion


LAUNCHER = '''CASE_PATH=${CASE_PATH:-demo.json}
BERNINI_CONFIG=${BERNINI_CONFIG:-model}
torchrun infer_multi_gpu.py --config "$BERNINI_CONFIG" --case "$CASE_PATH" --seed 42 --num_frames 81 --fps 16 --num_inference_steps 40 --omega_txt 4 --vit_txt_cfg 1.2
'''


def test_launcher_changes_only_three_numeric_options():
    actual = bernini.prepare_launcher(LAUNCHER, seed=7, num_frames=137, fps=25)
    assert actual == LAUNCHER.replace("--seed 42", "--seed 7").replace(
        "--num_frames 81", "--num_frames 137").replace("--fps 16", "--fps 25") + "\n"


@pytest.mark.parametrize("script", [LAUNCHER.replace("--num_frames 81", ""),
    LAUNCHER + " --fps 24", LAUNCHER.replace("--fps 16", "--fps 16.0"),
    "if true; then\n" + LAUNCHER + "fi\n",
    "run_demo() {\n" + LAUNCHER + "}\nrun_demo\nrun_demo\n",
    "while true; do\n" + LAUNCHER + "done\n"])
def test_unknown_launcher_shape_fails(script):
    with pytest.raises(ValueError):
        bernini.prepare_launcher(script, seed=7, num_frames=137, fps=25)


def test_probe_uses_decoded_count_and_rational_fps(monkeypatch):
    def probe(args, **kwargs):
        assert "-count_frames" in args
        return SimpleNamespace(stdout=json.dumps({"streams": [{"nb_read_frames":"138", "nb_frames":"81",
            "avg_frame_rate":"24000/1001", "time_base":"1/24000", "width":640, "height":480}],
            "frames":[{"best_effort_timestamp":i*1001} for i in range(138)]}))
    monkeypatch.setattr(timeline.subprocess, "run", probe)
    inspected = timeline.inspect_video_timeline(Path("source.mp4"))
    assert inspected == source(138, 24000, 1001)


def test_probe_rejects_irregular_pts_despite_matching_average_fps(monkeypatch):
    payload = {"streams":[{"nb_read_frames":"5", "avg_frame_rate":"25/1",
        "time_base":"1/1000", "width":640, "height":480}],
        "frames":[{"best_effort_timestamp":pts} for pts in [0,20,80,100,160]]}
    monkeypatch.setattr(timeline.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(payload)))
    with pytest.raises(ValueError, match="cadence"):
        timeline.inspect_video_timeline(Path("variable.mp4"))


def test_prepare_padding_has_only_tail_repeat_and_no_resampling(tmp_path, monkeypatch):
    plan = bernini.plan_bernini_timeline(source())
    calls = []
    monkeypatch.setattr(bernini, "inspect_video_timeline", lambda p: source(141))
    monkeypatch.setattr(timeline.subprocess, "run", lambda args, **kw: calls.append(args))
    original = tmp_path / "original.mp4"
    original.write_bytes(b"unchanged")
    case = tmp_path / "case"
    case.mkdir()
    result = bernini.prepare_generation_input(original, case, plan)
    assert result == case / "bernini_input.mp4"
    cmd = calls[0]
    assert cmd[cmd.index("-vf") + 1] == "tpad=stop_mode=clone:stop=3,settb=expr=1/25,setpts=N*1"
    assert "-shortest" not in cmd and "-t" not in cmd
    assert original.read_bytes() == b"unchanged"


def test_direct_input_for_aligned_integer_source(tmp_path):
    path = tmp_path / "source.mp4"
    assert bernini.prepare_generation_input(path, tmp_path, bernini.plan_bernini_timeline(source(137))) == path


def test_trim_only_tail_and_restore_exact_rational(tmp_path, monkeypatch):
    plan = bernini.plan_bernini_timeline(source(138, 24000, 1001))
    raw = tmp_path / "replacement_raw.mp4"
    final = tmp_path / "replacement.mp4"
    calls = []
    monkeypatch.setattr(bernini, "inspect_video_timeline", lambda p: source(141,24) if p == raw else plan.source)
    monkeypatch.setattr(timeline.subprocess, "run", lambda args, **kw: calls.append(args))
    before, after = bernini.normalize_output(raw, final, plan)
    assert before.frame_count == 141 and after.frame_count == 138
    cmd = calls[0]
    assert cmd[cmd.index("-vf") + 1] == "trim=end_frame=138,settb=expr=1/24000,setpts=N*1001"
    assert cmd[cmd.index("-r") + 1] == "24000/1001"
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert "-shortest" not in cmd and "-t" not in cmd


def test_normalization_promotes_without_reencode_when_already_exact(tmp_path, monkeypatch):
    plan = bernini.plan_bernini_timeline(source(81,16))
    raw, final = tmp_path / "raw.mp4", tmp_path / "replacement.mp4"
    raw.write_bytes(b"valid mock mp4 bytes")
    monkeypatch.setattr(bernini, "inspect_video_timeline", lambda p: plan.source)
    monkeypatch.setattr(timeline.subprocess, "run", lambda *a, **kw: pytest.fail("unneeded reencode"))
    bernini.normalize_output(raw, final, plan)
    assert final.read_bytes() == b"valid mock mp4 bytes"


@pytest.mark.parametrize("changed", [{"frame_count":137}, {"fps":16,"fps_num":16},
                                   {"duration_seconds":9}])
def test_final_timeline_mismatch_is_fatal(changed):
    with pytest.raises(ValueError):
        timeline.validate_matching_timeline(source(), replace(source(), **changed))


@pytest.mark.parametrize("bad", ["count", "fps"])
def test_pipeline_never_publishes_mismatching_output(tmp_path, monkeypatch, bad):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"source")
    case = tmp_path / "case"
    events = []
    qwen = SimpleNamespace(model_path=tmp_path, describe=lambda p: events.append("describe") or "person",
        invent=lambda s: events.append("invent") or "appearance", close=lambda: events.append("close"))
    def run(case_path, seed, *, plan):
        events.append("bernini")
        assert plan.internal_frame_count == 141
    adapter = SimpleNamespace(config=tmp_path, validate=lambda: None, run=run)
    monkeypatch.setattr(pipeline, "inspect_video_timeline", lambda p: source())
    monkeypatch.setattr(pipeline, "prepare_generation_input", lambda v,o,p: v)
    monkeypatch.setattr(pipeline, "extract_frame_zero", lambda v,p: p.write_bytes(b"reference"))
    monkeypatch.setattr(bernini, "inspect_video_timeline", lambda p:
        source(141) if p.name == "replacement_raw.mp4" else (source(137) if bad == "count" else source(138,16)))
    monkeypatch.setattr(timeline.subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(ValueError):
        pipeline.run_cases([{"output_dir":str(case), "target_video_path":str(path)}], qwen=qwen, bernini=adapter, seed=42)
    assert events == ["describe", "invent", "close", "bernini"]
    assert not (case / "manifest.json").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg tools unavailable")
def test_real_ffmpeg_tail_padding_trim_and_fractional_timing(tmp_path):
    import cv2
    import numpy as np

    original = tmp_path / "original.mp4"
    # Each frame is a distinct gray level; verify decoded order after real FFmpeg.
    pixels = b"".join(np.full((16, 16, 3), i, dtype=np.uint8).tobytes() for i in range(138))
    subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "16x16", "-r", "24000/1001", "-i", "pipe:0", "-c:v", "libx264",
        "-crf", "0", "-pix_fmt", "yuv420p", str(original)], input=pixels, check=True)
    before = original.read_bytes()
    plan = bernini.plan_bernini_timeline(timeline.inspect_video_timeline(original))
    assert plan.source.rate.numerator == 24000 and plan.source.rate.denominator == 1001
    padded = bernini.prepare_generation_input(original, tmp_path, plan)

    def levels(path):
        capture = cv2.VideoCapture(str(path))
        result = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    return result
                result.append(float(frame.mean()))
        finally:
            capture.release()

    real = levels(original)
    internal = levels(padded)
    assert len(internal) == 141
    assert np.allclose(internal[:138], real, atol=1)
    assert np.allclose(internal[138:], [real[-1]] * 3, atol=1)
    final = tmp_path / "replacement.mp4"
    _, result = bernini.normalize_output(padded, final, plan)
    assert result.frame_count == 138 and result.rate == plan.source.rate
    assert np.allclose(levels(final), real, atol=3)
    assert original.read_bytes() == before
