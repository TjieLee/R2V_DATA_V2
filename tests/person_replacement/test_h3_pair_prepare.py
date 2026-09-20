from pathlib import Path

import pytest

from r2v_data_v2.person_replacement.h3_pair_state import (
    atomic_json,
    failure_count,
    inventory,
    phase,
    read_json,
    skipped,
)
from r2v_data_v2.person_replacement.timeline import VideoTimeline

PASSTHROUGH_AUDIO = {"reference_audio_normalized":False,"reference_audio_source_channels":None,
                     "reference_audio_target_channels":None}


def passthrough_audio(module, monkeypatch):
    monkeypatch.setattr(module,"ensure_h3_reference",lambda source,directory:(source,dict(PASSTHROUGH_AUDIO)))


def make_cases(tmp_path, count, clips):
    cases = []
    for i in range(count):
        video = clips/f"{i}.mp4"
        video.touch()
        cases.append({"case_id":str(i),"directory":str(tmp_path/f"case{i}"),
                      "row":{"video_path":video.name},"row_sha256":str(i),"source_index":i})
    return cases


class Qwen:
    fail_on = "1.mp4"  # the deterministic bad sample used by the partition test

    def __init__(self, path=None):
        self.calls = CALLS

    def _load(self):
        CALLS.append("load")

    def describe_two_with_performance(self, video):
        CALLS.append(video.name)
        if getattr(self,"fail_on",None) == video.name:
            raise ValueError("bad case")
        return ("source 1","source 2","holding and handling a hockey stick",
                "holding a black clipboard with both hands","They cross.")

    def invent_two(self, *args):
        return "replacement 1","replacement 2"

    def close(self):
        CALLS.append("close")


CALLS = []


@pytest.fixture(autouse=True)
def clear_calls():
    CALLS.clear()
    yield


def test_prepare_partition_reuses_markers_isolates_failures_and_loads_once(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,6,clips)
    atomic_json(Path(cases[0]["directory"])/"generation/manifest.json",{})
    atomic_json(Path(cases[2]["directory"])/"preparation/prepared.json",{})
    config = {"cases":cases,"clips_root":str(clips),"qwen_model":str(tmp_path),"seed":42,"pair_id":0,
              "identity":"identity","limits":{str(i):{"prepare":2,"generate":2} for i in range(6)}}
    passthrough_audio(module,monkeypatch)
    monkeypatch.setattr(module,"inspect_video_timeline",lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    module.prepare_partition(config,0,lambda _:Qwen())
    assert CALLS == ["load","4.mp4","close"]
    CALLS.clear()
    module.prepare_partition(config,1,lambda _:Qwen())
    assert CALLS == ["load","1.mp4","1.mp4","3.mp4","5.mp4","close"]
    assert failure_count(cases[1],"prepare") == 2
    assert phase(cases[3]) == phase(cases[4]) == phase(cases[5]) == "generate"
    CALLS.clear()
    module.prepare_partition(config,1,lambda _:Qwen())
    assert CALLS == []  # exhausted and prepared skip even model construction


def test_interrupted_prepare_resumes_without_repeating_committed_qwen(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,3,clips)
    config = {"cases":cases,"clips_root":str(clips),"qwen_model":str(tmp_path),"seed":42,"pair_id":0,
              "identity":"identity","limits":{str(i):{"prepare":2,"generate":2} for i in range(3)}}
    passthrough_audio(module,monkeypatch)
    monkeypatch.setattr(module,"inspect_video_timeline",lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    interrupted = True
    seen = []

    class Interrupting(Qwen):
        def describe_two_with_performance(self, video):
            seen.append(video.name)
            if interrupted and video.name == "2.mp4":
                raise KeyboardInterrupt
            return super().describe_two_with_performance(video)

    with pytest.raises(KeyboardInterrupt):
        module.prepare_partition(config,0,lambda _:Interrupting())
    before = (tmp_path/"case0/preparation/prepared.json").read_bytes()
    interrupted = False
    seen.clear()
    module.prepare_partition(config,0,lambda _:Interrupting())
    assert seen == ["2.mp4"]
    assert (tmp_path/"case0/preparation/prepared.json").read_bytes() == before


def test_short_source_is_durably_skipped_without_qwen(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    config = {"cases":cases,"clips_root":str(clips),"qwen_model":str(tmp_path),"seed":42,"pair_id":0,
              "identity":"identity","limits":{"0":{"prepare":2,"generate":2}}}
    passthrough_audio(module,monkeypatch)
    monkeypatch.setattr(module,"inspect_video_timeline",lambda _:VideoTimeline(118,25,25,1,1920,1080,4.9))
    built = []
    module.prepare_partition(config,0,lambda path:built.append(path) or Qwen())
    assert built == []  # Qwen is never constructed for an ineligible input
    assert CALLS == []
    marker = skipped(cases[0])
    assert marker["reason"] == "source_duration_below_5s"
    assert marker["minimum_source_duration_seconds"] == 5.0
    assert marker["source_timeline"]["duration_seconds"] == 4.9
    assert phase(cases[0]) == "skipped"
    assert failure_count(cases[0],"prepare") == 0
    assert not (Path(cases[0]["directory"])/"preparation/prepared.json").exists()
    counts = inventory(cases,config["limits"])
    assert counts["skipped"] == 1 and counts["unprepared"] == 0
    assert counts["exhausted_prepare"] == counts["exhausted_generate"] == 0
    module.prepare_partition(config,0,lambda path:built.append(path) or Qwen())
    assert built == [] and phase(cases[0]) == "skipped"  # resume keeps the skip


def test_five_second_source_enters_preparation(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    config = {"cases":cases,"clips_root":str(clips),"qwen_model":str(tmp_path),"seed":42,"pair_id":0,
              "identity":"identity","limits":{"0":{"prepare":2,"generate":2}}}
    passthrough_audio(module,monkeypatch)
    monkeypatch.setattr(module,"inspect_video_timeline",lambda _:VideoTimeline(125,25,25,1,1920,1080,5.0))
    module.prepare_partition(config,0,lambda _:Qwen())
    assert CALLS[0] == "load"
    assert phase(cases[0]) == "generate"
    prepared = read_json(Path(cases[0]["directory"])/"preparation/prepared.json")
    assert prepared["source_performance_1"] == "holding and handling a hockey stick"
    assert prepared["source_performance_2"] == "holding a black clipboard with both hands"
    directory = Path(cases[0]["directory"])/"preparation"
    assert (directory/"source_performance_1.txt").read_text().strip() == prepared["source_performance_1"]
    assert (directory/"source_performance_2.txt").read_text().strip() == prepared["source_performance_2"]
    assert prepared["h3_reference_source"] == prepared["source"]  # stereo passthrough


def test_planner_ineligible_source_is_skipped_not_failed(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    config = {"cases":cases,"clips_root":str(clips),"qwen_model":str(tmp_path),"seed":42,"pair_id":0,
              "identity":"identity","limits":{"0":{"prepare":2,"generate":2}}}
    passthrough_audio(module,monkeypatch)
    monkeypatch.setattr(module,"inspect_video_timeline",lambda _:VideoTimeline(500,25,25,1,1920,1080,20.0))
    built = []
    module.prepare_partition(config,0,lambda path:built.append(path) or Qwen())
    assert built == []
    marker = skipped(cases[0])
    assert marker["reason"] == "h3_timeline_ineligible"
    assert marker["detail"]
    assert phase(cases[0]) == "skipped" and failure_count(cases[0],"prepare") == 0


def test_inspect_failure_is_deterministic_skip(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    config = {"cases":cases,"clips_root":str(clips),"qwen_model":str(tmp_path),"seed":42,"pair_id":0,
              "identity":"identity","limits":{"0":{"prepare":2,"generate":2}}}
    passthrough_audio(module,monkeypatch)

    def broken(_):
        raise ValueError("Unsupported variable/invalid frame cadence")

    monkeypatch.setattr(module,"inspect_video_timeline",broken)
    module.prepare_partition(config,0,lambda _:Qwen())
    assert skipped(cases[0])["reason"] == "h3_timeline_ineligible"
    assert failure_count(cases[0],"prepare") == 0


def test_multichannel_reference_is_recorded_for_h3(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = make_cases(tmp_path,1,clips)
    config = {"cases":cases,"clips_root":str(clips),"qwen_model":str(tmp_path),"seed":42,"pair_id":0,
              "identity":"identity","limits":{"0":{"prepare":2,"generate":2}}}
    monkeypatch.setattr(module,"inspect_video_timeline",lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    seen = {}

    def fake_reference(source, directory):
        seen["source"] = source
        target = Path(directory)/"h3_reference_stereo.mp4"
        return target,{"reference_audio_normalized":True,"reference_audio_source_channels":6,
                       "reference_audio_target_channels":2}

    monkeypatch.setattr(module,"ensure_h3_reference",fake_reference)
    module.prepare_partition(config,0,lambda _:Qwen())
    prepared = read_json(Path(cases[0]["directory"])/"preparation/prepared.json")
    assert prepared["source"] == str(clips/"0.mp4")  # original provenance preserved
    assert prepared["h3_reference_source"] == str(
        Path(cases[0]["directory"])/"preparation/h3_reference_stereo.mp4")
    assert prepared["reference_audio_normalized"] is True
    assert prepared["reference_audio_source_channels"] == 6
    assert prepared["reference_audio_target_channels"] == 2
