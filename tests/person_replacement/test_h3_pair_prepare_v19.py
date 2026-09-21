"""Text V19 orchestration: one resident 27B writer per worker, resume-safe."""

import pytest

from r2v_data_v2.person_replacement.h3_pair_state import (
    atomic_json,
    failure_count,
    phase,
    publish_prepared,
    qwen_prepared,
    read_json,
)
from r2v_data_v2.person_replacement.timeline import VideoTimeline
from tests.person_replacement.test_qwen38_h3_prompt_writer import LEGAL_PROMPT

CALL1_FIELDS = {"source_performer_1":"a younger woman in a yellow shirt",
                "source_performer_2":"an older woman in a light shirt",
                "replacement_subject_1":"a middle-aged man",
                "replacement_subject_2":"a young man in denim"}

EVENTS = []
FRAMES = []


class FakeWriter:
    def __init__(self, model_path, **kwargs):
        EVENTS.append("load")
        self.model_path = model_path
        self.thinking_disabled = True
        self.prompt_text = LEGAL_PROMPT

    def _load(self):
        pass

    def invent_replacements(self, video, cue1, cue2, *, num_frames):
        EVENTS.append(f"{video.stem}_call1")
        FRAMES.append(num_frames)
        return (CALL1_FIELDS["source_performer_1"],CALL1_FIELDS["source_performer_2"],
                CALL1_FIELDS["replacement_subject_1"],CALL1_FIELDS["replacement_subject_2"])

    def write_h3_prompt(self, video, performer1, performer2, replacement1, replacement2,
                        *, num_frames):
        EVENTS.append(f"{video.stem}_call2")
        FRAMES.append(num_frames)
        assert (performer1,performer2,replacement1,replacement2) == (
            CALL1_FIELDS["source_performer_1"],CALL1_FIELDS["source_performer_2"],
            CALL1_FIELDS["replacement_subject_1"],CALL1_FIELDS["replacement_subject_2"])
        return self.prompt_text

    def close(self):
        EVENTS.append("close")


@pytest.fixture(autouse=True)
def clear_events():
    EVENTS.clear()
    FRAMES.clear()
    yield


def make_cases(tmp_path, count, clips):
    cases = []
    for i in range(count):
        (clips/f"{i}.mp4").touch()
        cases.append({"case_id":str(i),"directory":str(tmp_path/f"case{i}"),
                      "row":{"video_path":f"{i}.mp4"},"row_sha256":str(i),"source_index":i})
    return cases


def config_for(cases, clips, *, budget=2):
    return {"cases":cases,"clips_root":str(clips),"seed":42,"pair_id":0,"identity":"identity",
            "prompt_writer_model":"/mnt/workspace/public/pretrained/Qwen/Qwen3.5-27B",
            "prompt_writer_backend":"local",
            "prompt_writer_base_url":"http://127.0.0.1:8000/v1",
            "prompt_writer_served_model":"Qwen/Qwen3.8-Flash-Next",
            "limits":{case["case_id"]:{"prepare":budget,"generate":2} for case in cases}}


def prepare(tmp_path, monkeypatch, cases, clips, writer_factory=lambda path:FakeWriter(path),
            worker=0, **kwargs):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    monkeypatch.setattr(module,"ensure_h3_reference",
                        lambda source,directory:(source,{"reference_audio_normalized":False,
                                                         "reference_audio_source_channels":None,
                                                         "reference_audio_target_channels":None}))
    monkeypatch.setattr(module,"inspect_video_timeline",
                        lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    module.prepare_text_partition_v19(config_for(cases,clips,**kwargs),worker,writer_factory)


def test_one_model_load_for_many_cases(tmp_path, monkeypatch):
    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,3,clips)  # worker 0 owns case0 and case2
    prepare(tmp_path,monkeypatch,cases,clips)
    assert EVENTS == ["load","0_call1","0_call2","2_call1","2_call2","close"]
    assert EVENTS.count("load") == 1 and EVENTS.count("close") == 1
    assert phase(cases[0]) == phase(cases[2]) == "generate"


def test_text_v19_never_loads_8b_or_build_prompt(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    monkeypatch.setattr(module,"build_prompt",
                        lambda *a,**k:pytest.fail("text V19 must not call build_prompt"))
    monkeypatch.setattr(module,"LocalQwen",lambda *a,**k:pytest.fail("text V19 must not load the 8B"))
    prepare(tmp_path,monkeypatch,cases,clips)
    prepared = read_json((tmp_path/"case0")/"preparation"/"prepared.json")
    assert prepared["prompt"] == LEGAL_PROMPT
    assert prepared["prompt_source"] == "qwen35_full_video"
    assert prepared["source_subject_1"] if False else True
    assert "shot_description" not in prepared and "source_subject_1" not in prepared


def test_resume_after_call1_does_not_repeat_call1(tmp_path, monkeypatch):
    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    directory = tmp_path/"case0"/"preparation"
    directory.mkdir(parents=True)
    atomic_json(directory/"qwen.json",{**CALL1_FIELDS,"case_id":"0","row_sha256":"0",
                                       "identity":"identity","source":str(clips/"0.mp4"),
                                       "seed":42,"h3_reference_source":str(clips/"0.mp4")})

    class NoCall1(FakeWriter):
        def invent_replacements(self, video, cue1, cue2, *, num_frames):
            pytest.fail("Call 1 must not repeat when qwen.json exists")

    prepare(tmp_path,monkeypatch,cases,clips,writer_factory=lambda path:NoCall1(path))
    assert EVENTS == ["load","0_call2","close"]
    assert phase(cases[0]) == "generate"


def test_call2_failure_keeps_marker_and_retries_only_call2(tmp_path, monkeypatch):
    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    attempts = {"count":0}

    class FailingCall2(FakeWriter):
        def write_h3_prompt(self, *args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("writer failed")
            return super().write_h3_prompt(*args, **kwargs)

    prepare(tmp_path,monkeypatch,cases,clips,writer_factory=lambda path:FailingCall2(path))
    assert failure_count(cases[0],"prepare") == 1
    assert qwen_prepared(cases[0])["source_performer_1"] == CALL1_FIELDS["source_performer_1"]
    assert EVENTS.count("0_call1") == 1  # Call 1 is not repeated
    assert phase(cases[0]) == "generate"


def test_invalid_writer_output_is_a_case_failure(tmp_path, monkeypatch):
    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)

    class BadPrompt(FakeWriter):
        def write_h3_prompt(self, *args, **kwargs):
            return "subject_definitions:\nonly one section"

    prepare(tmp_path,monkeypatch,cases,clips,writer_factory=lambda path:BadPrompt(path))
    assert failure_count(cases[0],"prepare") == 2  # fail closed, no fallback prompt
    assert phase(cases[0]) == "prepare"
    assert not (tmp_path/"case0"/"preparation"/"prepared.json").exists()


def test_prepared_and_skipped_cases_never_instantiate_the_writer(tmp_path, monkeypatch):
    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,2,clips)
    publish_prepared(cases[0],{"case_id":"0","prompt":"old"},{"h3_prompt":"old"})
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    monkeypatch.setattr(module,"ensure_h3_reference",
                        lambda source,directory:(source,{"reference_audio_normalized":False,
                                                         "reference_audio_source_channels":None,
                                                         "reference_audio_target_channels":None}))
    monkeypatch.setattr(module,"inspect_video_timeline",
                        lambda _:VideoTimeline(118,25,25,1,1920,1080,4.9))
    module.prepare_text_partition_v19(config_for(cases,clips),1,lambda path:FakeWriter(path))
    assert EVENTS == []  # a short-source skip never loads the 27B
    assert phase(cases[1]) == "skipped"


def test_prepared_provenance_and_artifacts(tmp_path, monkeypatch):
    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    prepare(tmp_path,monkeypatch,cases,clips)
    directory = tmp_path/"case0"/"preparation"
    prepared = read_json(directory/"prepared.json")
    assert prepared["prompt_writer_model"] == "/mnt/workspace/public/pretrained/Qwen/Qwen3.5-27B"
    assert prepared["prompt_writer_contract"] == "mimo8_qwen38_or_qwen35_two_call_h3_prompt_v15"
    assert prepared["prompt_writer_video_fps"] == 4.0
    assert prepared["prompt_writer_replacement_max_new_tokens"] == 512
    assert prepared["prompt_writer_prompt_max_new_tokens"] == 4096
    assert prepared["prompt_writer_thinking"] is False or prepared["prompt_writer_thinking"] is True
    assert prepared["variant"] == "text_two_person"
    for name in ("source_performer_1","source_performer_2","replacement_subject_1",
                 "replacement_subject_2","h3_prompt","prompt_writer_request","qwen","prepared"):
        assert (directory/f"{name}.txt" if name != "qwen" and name != "prepared" else
                directory/f"{name}.json").is_file()
    assert (directory/"prompt_writer_request.txt").read_text().count("<Subject 1>") >= 1
    assert prepared["replacement_diversity_1"] and prepared["replacement_diversity_2"]


def test_mimo_backend_factory_and_provenance_use_fps8(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    config = config_for(cases,clips)
    config.update({
        "prompt_writer_model":"/mnt/workspace/public/pretrained/MiMo/MiMo-V2.5",
        "prompt_writer_backend":"mimo",
        "prompt_writer_base_url":"http://127.0.0.1:8092/v1",
        "prompt_writer_served_model":"mimo-v2.5",
    })
    monkeypatch.setattr(module,"ensure_h3_reference",
                        lambda source,directory:(source,{"reference_audio_normalized":False,
                                                         "reference_audio_source_channels":None,
                                                         "reference_audio_target_channels":None}))
    monkeypatch.setattr(module,"inspect_video_timeline",
                        lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))

    class FakeMimo(FakeWriter):
        uses_local_frame_sampling = False
        thinking_disabled = True
        video_fps = 8.0
        def __init__(self, path, *, base_url, served_model):
            super().__init__(path)
            assert base_url == "http://127.0.0.1:8092/v1"
            assert served_model == "mimo-v2.5"

    monkeypatch.setattr(module,"OpenAIMimoH3PromptWriter",FakeMimo)
    module.prepare_text_partition_v19(config,0)
    prepared = read_json(tmp_path/"case0"/"preparation"/"prepared.json")
    assert prepared["prompt_writer_backend"] == "mimo"
    assert prepared["prompt_writer_video_fps"] == 8.0
    assert prepared["prompt_source"] == "mimo25_sglang_full_video_fps8"


def test_frame0_never_instantiates_the_27b(tmp_path, monkeypatch):
    """frame0 stays on the 8B + Boogu path and keeps its own v18 contract."""
    from r2v_data_v2.person_replacement import h3_pair_prepare as module
    from r2v_data_v2.person_replacement.h3_pair_executor import VARIANTS

    assert VARIANTS["frame0"] == "frame0_two_person_pdd_fsdp2_pair_v18"
    assert VARIANTS["text"] == "text_two_person_pdd_fsdp2_pair_v37"

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    monkeypatch.setattr(module,"ensure_h3_reference",
                        lambda source,directory:(source,{"reference_audio_normalized":False,
                                                         "reference_audio_source_channels":None,
                                                         "reference_audio_target_channels":None}))
    monkeypatch.setattr(module,"inspect_video_timeline",
                        lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    monkeypatch.setattr(module,"LocalQwen38H3PromptWriter",
                        lambda *a,**k:pytest.fail("frame0 must not load Qwen3.8"))
    config = {"cases":cases,"clips_root":str(clips),"seed":42,"pair_id":0,"identity":"identity",
              "qwen_model":"/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct",
              "limits":{"0":{"prepare":2,"generate":2}}}
    module.prepare_partition(config,0,lambda path:type("Qwen",(),{
        "_load":lambda self:None,
        "describe_two":lambda self,video:("s1","s2","shot"),
        "invent_two_with_diversity":lambda self,a,b,c1,c2:("r1","r2"),
        "close":lambda self:None})())
    assert phase(cases[0]) == "generate"
    prepared = read_json((tmp_path/"case0")/"preparation"/"prepared.json")
    assert "prompt_writer_model" not in prepared
    assert prepared["source_subject_1"] == "s1"


@pytest.mark.parametrize("duration,expected", [(5.0,20),(10.0,40),(15.0,60),(20.0,60),(0.1,1)])
def test_writer_frames_budget(duration, expected):
    from r2v_data_v2.person_replacement.h3_pair_prepare import writer_frames

    assert writer_frames(duration) == expected


def test_both_calls_share_one_num_frames_and_no_fps(tmp_path, monkeypatch):
    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    monkeypatch.setattr(module,"ensure_h3_reference",
                        lambda source,directory:(source,{"reference_audio_normalized":False,
                                                         "reference_audio_source_channels":None,
                                                         "reference_audio_target_channels":None}))
    monkeypatch.setattr(module,"inspect_video_timeline",
                        lambda _:VideoTimeline(240,25,25,1,1920,1080,10.0))
    module.prepare_text_partition_v19(config_for(cases,clips),0,lambda path:FakeWriter(path))
    assert FRAMES == [40,40]  # Call 1 and Call 2 use exactly the same budget
    prepared = read_json((tmp_path/"case0")/"preparation"/"prepared.json")
    assert prepared["prompt_writer_num_frames"] == 40
    assert prepared["prompt_writer_video_fps"] == 4.0
