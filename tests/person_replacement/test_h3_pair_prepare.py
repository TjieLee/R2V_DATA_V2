from pathlib import Path

import pytest

from r2v_data_v2.person_replacement.h3_pair_state import (
    atomic_json,
    failure_count,
    phase,
)
from r2v_data_v2.person_replacement.timeline import VideoTimeline


def test_prepare_partition_reuses_markers_isolates_failures_and_loads_once(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = []
    for i in range(6):
        video = clips/f"{i}.mp4"
        video.touch()
        cases.append({"case_id":str(i),"directory":str(tmp_path/f"case{i}"),
                      "row":{"video_path":video.name},"row_sha256":str(i),"source_index":i})
    atomic_json(Path(cases[0]["directory"])/"generation/manifest.json",{})
    atomic_json(Path(cases[2]["directory"])/"preparation/prepared.json",{})
    config = {"cases":cases,"clips_root":str(clips),"qwen_model":str(tmp_path),"seed":42,"pair_id":0,
              "identity":"identity","limits":{str(i):{"prepare":2,"generate":2} for i in range(6)}}
    calls = []
    class Qwen:
        def _load(self):
            calls.append("load")
        def describe_two(self, video):
            calls.append(video.name)
            if video.name == "1.mp4":
                raise ValueError("bad case")
            return "source 1","source 2","They cross."
        def invent_two(self,*args):
            return "replacement 1","replacement 2"
        def close(self):
            calls.append("close")
    monkeypatch.setattr(module,"inspect_video_timeline",lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    module.prepare_partition(config,0,lambda _:Qwen())
    assert calls == ["load","4.mp4","close"]
    calls.clear()
    module.prepare_partition(config,1,lambda _:Qwen())
    assert calls == ["load","1.mp4","1.mp4","3.mp4","5.mp4","close"]
    assert failure_count(cases[1],"prepare") == 2
    assert phase(cases[3]) == phase(cases[4]) == phase(cases[5]) == "generate"
    calls.clear()
    module.prepare_partition(config,1,lambda _:Qwen())
    assert calls == []  # exhausted and prepared skip even model construction


def test_interrupted_prepare_resumes_without_repeating_committed_qwen(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_prepare as module

    clips = tmp_path/"clips"
    clips.mkdir()
    cases = []
    for i in range(3):
        (clips/f"{i}.mp4").touch()
        cases.append({"case_id":str(i),"directory":str(tmp_path/f"case{i}"),
                      "row":{"video_path":f"{i}.mp4"},"row_sha256":str(i),"source_index":i})
    config = {"cases":cases,"clips_root":str(clips),"qwen_model":str(tmp_path),"seed":42,"pair_id":0,
              "identity":"identity","limits":{str(i):{"prepare":2,"generate":2} for i in range(3)}}
    interrupted = True
    seen = []
    class Qwen:
        def _load(self):
            pass
        def describe_two(self, video):
            seen.append(video.name)
            if interrupted and video.name == "2.mp4":
                raise KeyboardInterrupt
            return "one","two","They cross."
        def invent_two(self,*args):
            return "new one","new two"
        def close(self):
            pass
    monkeypatch.setattr(module,"inspect_video_timeline",lambda _:VideoTimeline(138,25,25,1,1920,1080,5.52))
    with pytest.raises(KeyboardInterrupt):
        module.prepare_partition(config,0,lambda _:Qwen())
    before = (tmp_path/"case0/preparation/prepared.json").read_bytes()
    interrupted = False
    seen.clear()
    module.prepare_partition(config,0,lambda _:Qwen())
    assert seen == ["2.mp4"]
    assert (tmp_path/"case0/preparation/prepared.json").read_bytes() == before
