import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from r2v_data_v2.person_replacement import h3_pipeline
from r2v_data_v2.person_replacement.h3_ref2va import (
    match_aspect_ratio,
    plan_h3_timeline,
)
from r2v_data_v2.person_replacement.timeline import VideoTimeline


def source():
    return VideoTimeline(138, 25.0, 25, 1, 1920, 1080, 5.52)


def test_native_timeline_not_source_normalized():
    plan = plan_h3_timeline(source())
    assert (plan.native_fps, plan.native_frame_count) == (24, 141)
    assert plan.requested_duration == 5.52
    assert plan.duration_drift_seconds == pytest.approx(141/24-5.52)
    assert plan_h3_timeline(replace(source(), duration_seconds=2)).native_frame_count == 124


@pytest.mark.parametrize("seconds", [1.9, 15.1, 15.0])
def test_duration_outside_reference_or_aligned_runtime_range_fails(seconds):
    with pytest.raises(ValueError):
        plan_h3_timeline(replace(source(), duration_seconds=seconds))


def test_aspect_ratio_no_crop():
    assert match_aspect_ratio(source()) == "16:9"
    with pytest.raises(ValueError):
        match_aspect_ratio(replace(source(), width=1500, height=1000))


@pytest.mark.parametrize("failed", ["lightx2v", "pdd", "preflight_lightx2v", "preflight_pdd",
                                    "bad_count", "bad_fps", "bad_size", None])
@pytest.mark.parametrize("reuse", [False, True])
def test_shared_preparation_and_backend_specific_manifest_gate(tmp_path, monkeypatch, failed, reuse):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"original")
    output = tmp_path / "case"
    events, prompts = [], []
    qwen = SimpleNamespace(describe=lambda v: events.append(("describe", v)) or "source",
        invent=lambda s: events.append(("invent", s)) or "replacement", close=lambda: events.append("close"))

    class Backend:
        def __init__(self, name):
            self.name = name
            self.model_root = tmp_path
            self.lora = tmp_path / name

        def validate(self):
            if failed == f"preflight_{self.name}":
                raise ValueError("bad checkpoint")

        def run(self, video_path, prompt, plan, aspect, directory, seed):
            assert "close" in events and video_path == video
            prompts.append(prompt)
            events.append(self.name)
            (directory / "raw.mp4").write_bytes(b"native")
            if self.name == failed:
                raise RuntimeError("backend failed")
            return {"inference_nfe": 8}

    monkeypatch.setattr(h3_pipeline, "extract_frame_zero", lambda v, p: p.write_bytes(v.read_bytes()))
    native = replace(source(), frame_count=141, fps=24., fps_num=24, duration_seconds=141/24,
                     width=1376, height=768)
    if failed == "bad_count":
        native = replace(native, frame_count=140)
    elif failed == "bad_fps":
        native = replace(native, fps=25., fps_num=25)
    elif failed == "bad_size":
        native = replace(native, width=1344)
    monkeypatch.setattr(h3_pipeline, "inspect_video_timeline", lambda p: source() if p == video else native)
    previous = tmp_path/"bernini_manifest.json"
    previous.write_text(json.dumps({"target_video_path":str(video), "source_person_description":"source",
                                   "replacement_person_description":"replacement"}))
    report = h3_pipeline.run_cases([{"case_id":"00209", "target_video_path":str(video),
        "output_dir":str(output)}], qwen=qwen, backends=[Backend("lightx2v"), Backend("pdd")], seed=42,
        descriptions_from=previous if reuse else None)
    assert len(prompts) == (1 if str(failed).startswith("preflight_") else 2)
    assert all(prompt is prompts[0] for prompt in prompts)
    if reuse:
        assert events[0] == "close" and not any(isinstance(event, tuple) for event in events)
    else:
        assert events[:3] == [("describe",video), ("invent","source"), "close"]
    for name in ["lightx2v", "pdd"]:
        path = output / name / "manifest.json"
        assert path.exists() == (name != failed and failed != f"preflight_{name}" and not str(failed).startswith("bad_"))
        if path.exists():
            manifest = json.loads(path.read_text())
            assert manifest["production_timeline_compatible"] is False
            assert manifest["target_video_path"] == str(video)
            assert manifest["output_frame_count"] == 141
    assert len(report["failures"]) == (2 if str(failed).startswith("bad_") else int(bool(failed)))
    assert (output / "reference_frame0.jpg").read_bytes() == b"original"


def test_dry_run_no_models_workers_or_directories(tmp_path, monkeypatch, capsys):
    from tools.person_replacement import run_h3_ref2va_pilot as cli

    out = tmp_path / "outputs"
    monkeypatch.setattr(cli, "select_cases", lambda *a, **k: [{"target_video_path":"source.mp4", "output_dir":str(out)}])
    monkeypatch.setattr(cli, "inspect_video_timeline", lambda p: source())
    monkeypatch.setattr(cli, "LocalQwen", lambda *a: pytest.fail("loaded Qwen"))
    monkeypatch.setattr(cli, "run_cases", lambda *a, **k: pytest.fail("launched worker"))
    assert cli.main(["--input-jsonl",str(tmp_path/"input.jsonl"), "--clips-root",str(tmp_path),
                     "--output-root",str(out), "--dry-run"]) == 0
    assert not out.exists()
    data = json.loads(capsys.readouterr().out)
    assert data["cases"][0]["h3_timeline"]["native_frame_count"] == 141
    assert set(data["contracts"]) == {"lightx2v", "pdd"}


def test_reuse_descriptions_requires_matching_original_source(tmp_path):
    from r2v_data_v2.person_replacement.h3_pipeline import read_descriptions

    video = tmp_path/"original.mp4"
    manifest = tmp_path/"bernini_manifest.json"
    manifest.write_text(json.dumps({"target_video_path":str(video),
        "source_person_description":"source", "replacement_person_description":"replacement"}))
    assert read_descriptions(manifest, video) == ("source", "replacement")
    with pytest.raises(ValueError, match="source"):
        read_descriptions(manifest, tmp_path/"different.mp4")


def test_all_invalid_backends_do_not_load_qwen_or_write_case(tmp_path):
    def fail():
        raise ValueError("missing local weights")
    backend = SimpleNamespace(name="pdd", validate=fail)
    qwen = SimpleNamespace(close=lambda:None, describe=lambda *a:pytest.fail("Qwen must not load"))
    out = tmp_path/"output"
    result = h3_pipeline.run_cases([{"case_id":"one", "output_dir":str(out)}],
        qwen=qwen, backends=[backend], seed=42)
    assert result["manifests"] == [] and len(result["failures"]) == 1
    assert not out.exists()
