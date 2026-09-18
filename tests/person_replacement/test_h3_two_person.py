import json
import os
import re
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from r2v_data_v2.person_replacement import qwen3vl

DESCRIPTIONS = ("short hair and blue coat, initially left", "long hair and red coat, initially right",
                "<Subject 1> passes behind <Subject 2> while both hold a box; the camera pans right.")
REPLACEMENTS = ("older adult with curly hair and green jacket", "young adult with bob and tan shirt")


@pytest.mark.parametrize("frame0", [False, True])
def test_six_section_two_person_prompt(frame0):
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    prompt = build_prompt(*DESCRIPTIONS, *REPLACEMENTS, frame0=frame0)
    assert re.findall(r"^([a-z_]+):", prompt, re.MULTILINE) == [
        "subject_definitions", "summary", "retention_analysis", "detailed_description",
        "overall_soundscape", "non_diegetic_music"]
    assert "[Shot 1]" in prompt and "[Shot 2]" not in prompt
    assert re.search(r"(?<!<)\bSubject [12]\b(?!>)", prompt) is None
    assert ("<Picture 1>" in prompt) == frame0
    for value in (*DESCRIPTIONS[:2], *REPLACEMENTS, "<Subject 1>", "<Subject 2>",
                  "<Subject 3>", "<Subject 4>", "partially_preserved", "fully_preserved",
                  "<Video 1>"):
        assert value in prompt
    assert "<Subject 3> passes behind <Subject 4> while both hold a box" in prompt
    assert "do not swap the two target subjects" in prompt
    if frame0:
        assert "where visible" in prompt and "first-frame" in prompt


def test_boogu_prompt_both_bindings():
    from r2v_data_v2.person_replacement.h3_two_person import boogu_prompt

    prompt = boogu_prompt(*DESCRIPTIONS[:2], *REPLACEMENTS)
    for source, target in zip(DESCRIPTIONS[:2], REPLACEMENTS, strict=True):
        assert f"{source} -> {target}" in prompt
    for text in ("never swap", "pose", "contact", "background", "Do not add a missing person"):
        assert text in prompt


def test_two_person_qwen_calls_and_parser(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.touch()
    qwen = qwen3vl.LocalQwen(tmp_path)
    calls = []
    replies = iter(["SOURCE_SUBJECT_1: blue coat\nSOURCE_SUBJECT_2: red coat\nSHOT_DESCRIPTION: They cross.",
                    "REPLACEMENT_SUBJECT_1: green coat\nREPLACEMENT_SUBJECT_2: tan shirt"])
    def text(content):
        calls.append(content)
        return next(replies)
    monkeypatch.setattr(qwen, "_text", text)
    assert qwen.describe_two(video) == ("blue coat", "red coat", "They cross.")
    assert qwen.invent_two("blue coat", "red coat") == ("green coat", "tan shirt")
    assert calls[0][0] == {"type":"video", "path":str(video), "fps":4.0}
    assert "physical performer" in calls[0][1]["text"]


@pytest.mark.parametrize("reply", ["SOURCE_SUBJECT_1: blue coat", "SOURCE_SUBJECT_1: x\nSOURCE_SUBJECT_1: y",
                                   "SOURCE_SUBJECT_1: x\nSOURCE_SUBJECT_2: y\nSHOT_DESCRIPTION: [Shot 2] run"])
def test_missing_or_invalid_source_fields_fail(tmp_path, monkeypatch, reply):
    video = tmp_path / "v.mp4"
    video.touch()
    qwen = qwen3vl.LocalQwen(tmp_path)
    monkeypatch.setattr(qwen, "_text", lambda _: reply)
    with pytest.raises(ValueError):
        qwen.describe_two(video)


def test_face2_explicit_opt_in_uses_existing_resolver(tmp_path):
    from r2v_data_v2.person_replacement.pipeline import select_cases

    clips = tmp_path / "clips"
    clips.mkdir()
    video = clips / "shot.mp4"
    video.touch()
    source = tmp_path / "collected_face_2.jsonl"
    source.write_text(json.dumps({"video_path":str(video)}) + "\n")
    with pytest.raises(ValueError, match="not enabled"):
        select_cases(source, clips, tmp_path/"out", limit=1)
    cases = select_cases(source, clips, tmp_path/"out", limit=1, allow_two_person=True)
    assert Path(cases[0]["target_video_path"]) == video


def test_dry_run_reports_pair_without_backends(tmp_path, monkeypatch, capsys):
    from r2v_data_v2.person_replacement.timeline import VideoTimeline
    from tools.person_replacement import run_h3_pdd_two_person_ab as cli

    clips = tmp_path/"clips"
    clips.mkdir()
    video = clips / "clip.mp4"
    video.touch()
    source = tmp_path / "collected_face_2.jsonl"
    source.write_text(json.dumps({"video_path":str(video)})+"\n")
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run instantiated model backend")
    monkeypatch.setattr(cli, "LocalQwen", forbidden)
    monkeypatch.setattr(cli, "PDDBackend", forbidden)
    monkeypatch.setattr("r2v_data_v2.person_replacement.h3_two_person_pipeline.BooguSubprocessBackend", forbidden)
    monkeypatch.setattr(cli, "inspect_video_timeline", lambda _: VideoTimeline(138,25,25,1,1920,1080,5.52))
    out = tmp_path/"out"
    assert cli.main(["--input-jsonl",str(source),"--clips-root",str(clips),
                     "--output-root",str(out),"--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["variants"] == ["text_two_person", "frame0_two_person"]
    assert report["cases"][0]["output_size"] == [1376,768]
    assert not out.exists()


@pytest.mark.parametrize("boogu_fails", [False, True])
def test_paired_run_shares_inputs_and_decodes_only_frame_zero(tmp_path, monkeypatch, boogu_fails):
    from r2v_data_v2.person_replacement import h3_two_person_pipeline as module
    from r2v_data_v2.person_replacement.timeline import VideoTimeline
    from r2v_data_v2.v3.reference_edit_boogu import BooguWorkerConfig

    video = tmp_path/"original.mp4"
    video.write_bytes(b"original bytes")
    directory = tmp_path/"out"/"case"
    source = VideoTimeline(138,25,25,1,1920,1080,5.52)
    generated = replace(source, frame_count=141, fps=24, fps_num=24, duration_seconds=141/24, width=1376, height=768)
    monkeypatch.setattr(module, "inspect_video_timeline", lambda p: source if p == video else generated)
    events = []
    class Decoder:
        def decode_indices(self, path, indices):
            assert path == video and indices == [0]
            events.append("frame0")
            return [SimpleNamespace(image=Image.new("RGB", (1920,1080), "blue"))]
    monkeypatch.setattr(module, "OpenCvFrameDecoder", Decoder)
    class Qwen:
        model_path = tmp_path
        def describe_two(self, path):
            assert path == video
            events.append("describe")
            return DESCRIPTIONS
        def invent_two(self, *subjects):
            assert subjects == DESCRIPTIONS[:2]
            events.append("invent")
            return REPLACEMENTS
        def close(self):
            events.append("qwen_close")
    class Boogu:
        def start(self, **kwargs):
            events.append("boogu_start")
        def edit(self, **kwargs):
            events.append("boogu_edit")
            assert kwargs["source_rgb"].size == (1376,768)
            assert kwargs["seed"] == 42
            assert kwargs["thinking_enabled"] is False and kwargs["instruction_rewrite_enabled"] is False
            if boogu_fails:
                raise RuntimeError("generation failed")
            buf = BytesIO()
            Image.new("RGB", (1376,768), "green").save(buf,format="PNG")
            return SimpleNamespace(png_bytes=buf.getvalue(), worker_metadata={"steps":4})
        def close(self):
            events.append("boogu_close")
    monkeypatch.setattr(module, "BooguSubprocessBackend", lambda config: Boogu())
    runs = []
    class PDD:
        model_root = tmp_path
        lora = tmp_path/"weights"
        def validate(self):
            pass
        def run(self, path, prompt, plan, aspect, target, seed, *, reference_image=None):
            events.append("pdd")
            assert "qwen_close" in events
            if reference_image:
                assert events.index("boogu_close") < len(events)-1
            runs.append((path,plan,aspect,seed,reference_image))
            (target/"raw.mp4").write_bytes(b"native")
            return {"inference_nfe":8,"video_shift":12,"audio_shift":3}
    config = BooguWorkerConfig(python_executable=tmp_path/"python", code_root=tmp_path,
                              model_path=tmp_path, allowed_server_root=tmp_path)
    report = module.run_cases([{"case_id":"case","target_video_path":str(video),"output_dir":str(directory)}],
                              qwen=Qwen(), backend=PDD(), boogu_config=config, seed=99, variant="both")
    if boogu_fails:
        assert len(report["manifests"]) == 1 and len(report["failures"]) == 1
        assert len(runs) == 1 and runs[0][4] is None
        assert report["failures"][0]["variant"] == "frame0_two_person"
        assert not (directory/"frame0"/"manifest.json").exists()
        assert events[-1] == "boogu_close"
        return
    assert not report["failures"] and len(report["manifests"]) == 2
    assert runs[0][:4] == runs[1][:4] and runs[0][3] == 99
    assert runs[0][4] is None and runs[1][4] == directory/"frame0"/"repainted_frame0.png"
    assert events.count("boogu_edit") == events.count("frame0") == 1
    assert events.count("describe") == events.count("invent") == 1
    manifests = [json.loads(Path(path).read_text()) for path in report["manifests"]]
    assert [item["type"] for item in manifests[0]["references"]] == ["video"]
    assert [item["type"] for item in manifests[1]["references"]] == ["video","image"]
    assert manifests[1]["boogu"]["seed"] == 42
    assert video.read_bytes() == b"original bytes"
    assert (directory/"original.mp4").is_symlink()
    before = {p:p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        module.run_cases([{"case_id":"case","target_video_path":str(video),"output_dir":str(directory)}],
                         qwen=Qwen(), backend=PDD(), boogu_config=config, seed=99, variant="both")
    assert all(p.read_bytes() == data for p,data in before.items())


def test_missing_replacement_fields_fail(tmp_path, monkeypatch):
    qwen = qwen3vl.LocalQwen(tmp_path)
    monkeypatch.setattr(qwen,"_text",lambda _: "REPLACEMENT_SUBJECT_1: one")
    with pytest.raises(ValueError):
        qwen.invent_two("one", "two")


def test_runtime_caches_are_case_local_and_environment_restored(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement.h3_two_person_pipeline import (
        _runtime_environment,
    )

    monkeypatch.setenv("HF_HOME", "/readonly/model/cache")
    monkeypatch.setenv("PYTHONPATH", "/foreign/environment")
    original = os.environ.copy()
    with pytest.raises(RuntimeError), _runtime_environment(tmp_path):
        for name in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "TRITON_CACHE_DIR", "TMPDIR"):
            assert Path(os.environ[name]).is_relative_to(tmp_path)
        assert os.environ["HF_HUB_OFFLINE"] == "1" and "PYTHONPATH" not in os.environ
        raise RuntimeError("model failed")
    assert os.environ == original


def test_two_person_appearance_and_temporal_prompt_contract():
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt
    from r2v_data_v2.person_replacement.qwen3vl import (
        TWO_REPLACEMENT_PROMPT,
        TWO_SOURCE_PROMPT,
    )

    source_prompt = " ".join(TWO_SOURCE_PROMPT.split())
    assert "one concise natural description" in source_prompt
    assert "stable held object or pose may also be mentioned" in source_prompt
    assert "complete action sequence in SHOT_DESCRIPTION" in source_prompt
    assert "Do not omit, merge, reorder or invent actions" in source_prompt
    assert "Do not use person pronouns such as he, she" in source_prompt
    assert '"<Subject 1> raises <Subject 1>\'s right hand."' in source_prompt

    replacement_prompt = " ".join(TWO_REPLACEMENT_PROMPT.split())
    assert "clearly different from Source 1" in replacement_prompt
    assert "clearly different from Source 2" in replacement_prompt
    assert "Change several visible identity characteristics" in replacement_prompt
    assert "coherent realistic combination rather than forcing every category to differ" in replacement_prompt
    assert "concise and natural" in replacement_prompt

    shot = "<Subject 1> lifts the blue-and-white cup, drinks, then lowers it while <Subject 2> watches."
    source1 = "a bald man with a mustache in a dark blue robe seated in the foreground"
    source2 = "a shaved-head man in a light gray robe standing in the background"
    replacement1 = "A middle-aged woman with auburn hair in an olive-green jacket."
    replacement2 = "An adult man with short black hair in a cream linen shirt."
    prompt = build_prompt(source1, source2, shot, replacement1, replacement2)

    definitions = prompt.split("\n\nsummary:",1)[0]
    detailed = prompt.split("detailed_description:\n",1)[1].split("\n\noverall_soundscape:",1)[0]
    assert f"<Subject 1> is {source1} in <Video 1>." in definitions
    assert f"<Subject 2> is {source2} in <Video 1>." in definitions
    assert "<Subject 3> is a middle-aged woman with auburn hair in an olive-green jacket." in definitions
    assert "<Subject 4> is an adult man with short black hair in a cream linen shirt." in definitions
    assert "<Subject 3> replaces <Subject 1>'s visible appearance" in definitions
    assert "<Subject 4> replaces <Subject 2>'s visible appearance" in definitions
    assert "<Subject 3> follows <Subject 1>'s original performance" in definitions
    assert "<Subject 4> follows <Subject 2>'s original performance" in definitions

    assert "<Subject 3> lifts the blue-and-white cup, drinks, then lowers it while <Subject 4> watches." in detailed
    assert "<Subject 1> lifts the blue-and-white cup" not in detailed
    assert "<Subject 1> (appears in [Shot 1]): partially_preserved" in prompt
    assert "<Subject 2> (appears in [Shot 1]): partially_preserved" in prompt
    assert "<Subject 3> (appears in [Shot 1]): fully_preserved" in prompt
    assert "<Subject 4> (appears in [Shot 1]): fully_preserved" in prompt
    assert "<Video 1> (source video editing): fully_preserved" in prompt
    assert "(throughout [Shot 1])" not in prompt
    assert "The target video is an edited version of <Video 1>" in prompt

    for forbidden in ("Replacement 1", "Replacement 2", "->", "..",
                      "The edit has two simultaneous requirements", "full visible edited human appearance"):
        assert forbidden not in prompt
    for phrase in ("same order and timing", "non-person props", "held or touched objects",
                   "hand-object contact", "camera, background, lighting and composition",
                   "crossings, overlap or occlusion", "do not swap the two target subjects",
                   "Preserve source synchronized sound"):
        assert phrase in prompt
    assert "rather than forcing every category to differ" in TWO_REPLACEMENT_PROMPT


def test_two_person_video_sampling_only_and_short_motion_prompt(tmp_path, monkeypatch):
    video = tmp_path/"source.mp4"
    video.touch()
    client = qwen3vl.LocalQwen(tmp_path)
    calls = []
    def text(content):
        calls.append(content)
        return "SOURCE_SUBJECT_1: gray hair\nSOURCE_SUBJECT_2: red coat\nSHOT_DESCRIPTION: <Subject 1> raises a cup."
    monkeypatch.setattr(client,"_text",text)
    client.describe_two(video)
    client.describe(video)
    assert calls[0][0] == {"type":"video","path":str(video.resolve()),"fps":4.0}
    assert "fps" not in calls[1][0]
    prompt = calls[0][1]["text"]
    for phrase in ('<Subject 1>', '<Subject 2>', 'Never write bare "Subject 1" or "Subject 2"',
                   'brief hand, arm, head and object movements', 'every visible motion/state transition',
                   'genuinely motionless for the entire source sequence'):
        assert phrase in prompt


@pytest.mark.parametrize("shot", [
    "Subject 1 raises a cup while Subject 2 remains behind.",
    "<Subject 1> raises a cup while <Subject 2> remains behind.",
    "Subject 1 raises a cup while <Subject 2> remains behind.",
])
def test_shot_label_normalization_without_double_brackets(shot):
    from r2v_data_v2.person_replacement.h3_two_person import build_prompt

    prompt = build_prompt("gray hair","red coat",shot,"dark coat","short hair")
    detailed = prompt.split("detailed_description:\n",1)[1].split("\n\noverall_soundscape:",1)[0]
    assert "<Subject 3> raises a cup while <Subject 4> remains behind." in detailed
    assert "<<Subject" not in detailed
    assert "Subject 12" in build_prompt("a","b","Subject 12 stays.","c","d")
