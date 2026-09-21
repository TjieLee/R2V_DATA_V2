import json
from pathlib import Path

import pytest


def case(tmp_path):
    return {"case_id":"case", "directory":str(tmp_path/"case"), "source_index":0, "row":{"video_path":"a.mp4"}}


def test_markers_not_partial_artifacts_control_resume(tmp_path):
    from r2v_data_v2.person_replacement import h3_pair_state as state

    item = case(tmp_path)
    directory = tmp_path/"case"
    (directory/"preparation").mkdir(parents=True)
    (directory/"generation").mkdir()
    (directory/"preparation"/"h3_prompt.txt").write_text("partial")
    (directory/"generation"/"raw.mp4").write_bytes(b"partial")
    assert state.phase(item) == "prepare"
    state.atomic_json(directory/"preparation"/"prepared.json", {"case_id":"case"})
    assert state.phase(item) == "generate"
    state.atomic_json(directory/"generation"/"manifest.json", {})
    assert state.phase(item) == "done"


def test_failure_history_retry_budget_and_interruption(tmp_path):
    from r2v_data_v2.person_replacement import h3_pair_state as state

    item = case(tmp_path)
    # A killed attempt is not a failed case and remains retryable.
    state.begin_attempt(item,"prepare",{"pair_id":0})
    assert state.failure_count(item,"prepare") == 0
    for _ in range(2):
        attempt = state.begin_attempt(item,"prepare",{"pair_id":0})
        state.fail_attempt(item,attempt,ValueError("bad clip"))
    assert state.failure_count(item,"prepare") == 2
    assert not state.eligible(item,"prepare",2)
    assert state.retry_limit(item,"prepare",2,retry_failed=True) == 4
    assert state.eligible(item,"prepare",4)
    records = list((tmp_path/"case"/"failures").glob("*.json"))
    assert len(records) == 2
    assert [json.loads(p.read_text())["attempt"] for p in sorted(records)] == [2,3]


def test_publication_markers_last_and_done_never_overwritten(tmp_path, monkeypatch):
    from r2v_data_v2.person_replacement import h3_pair_state as state

    item = case(tmp_path)
    events = []
    real = state.atomic_json
    def write(path, payload):
        if path.name == "prepared.json":
            assert (path.parent/"h3_prompt.txt").read_text() == "prompt\n"
        if path.name == "manifest.json":
            assert (path.parent/"raw.mp4").read_bytes() == b"valid"
        events.append(path.name)
        real(path,payload)
    monkeypatch.setattr(state,"atomic_json",write)
    state.publish_prepared(item,{"case_id":"case","prompt":"prompt"},{"h3_prompt":"prompt"})
    assert events[-1] == "prepared.json"
    temporary = tmp_path/"case"/"tmp"/"video.mp4"
    temporary.parent.mkdir()
    temporary.write_bytes(b"valid")
    state.publish_generated(item,temporary,{"case_id":"case"},lambda p: events.append("validated"))
    assert events[-2:] == ["validated","manifest.json"]
    with pytest.raises(FileExistsError):
        state.publish_generated(item,temporary,{},lambda p: None)


def test_pair_shards_are_disjoint_and_partition_is_deterministic(tmp_path):
    from r2v_data_v2.person_replacement import h3_pair_state as state

    source = tmp_path/"data.jsonl"
    source.write_text("".join(json.dumps({"video_path":f"{i}.mp4"})+"\n" for i in range(7)))
    left = state.select_shard(source,tmp_path/"out",0,4)
    right = state.select_shard(source,tmp_path/"out",1,4)
    assert [x["source_index"] for x in left] == [0,1,2,3]
    assert [x["source_index"] for x in right] == [4,5,6]
    assert {x["case_id"] for x in left}.isdisjoint(x["case_id"] for x in right)
    assert [x["source_index"] for x in state.partition(left,0)] == [0,2]
    assert [x["source_index"] for x in state.partition(left,1)] == [1,3]
def test_skipped_is_durable_not_a_failure_and_not_done(tmp_path):
    from r2v_data_v2.person_replacement import h3_pair_state as state

    item = case(tmp_path)
    directory = tmp_path/"case"
    (directory/"preparation").mkdir(parents=True)
    assert state.phase(item) == "prepare"
    state.publish_skipped(item,{"case_id":"case","reason":"source_duration_below_5s",
                                "minimum_source_duration_seconds":5.0})
    assert state.phase(item) == "skipped"
    assert state.skipped(item)["reason"] == "source_duration_below_5s"
    assert not state.eligible(item,"prepare",2) and not state.eligible(item,"generate",2)
    with pytest.raises(FileExistsError):
        state.publish_skipped(item,{"case_id":"case","reason":"again"})
    with pytest.raises(ValueError):
        state.publish_skipped({"case_id":"x","directory":str(tmp_path/"other")},{"case_id":"x"})
    limits = {"case":{"prepare":2,"generate":2}}
    counts = state.inventory([item],limits)
    assert counts == {"done":0,"prepared":0,"unprepared":0,"skipped":1,
                      "exhausted_prepare":0,"exhausted_generate":0}
    assert state.finished(counts,1) is True  # settled, but not counted as done
    assert counts["done"] == 0
    assert state.finished(counts,2) is False


def test_skipped_and_done_together_finish_a_shard(tmp_path):
    from r2v_data_v2.person_replacement import h3_pair_state as state

    items = [case(tmp_path/f"case{index}") for index in range(3)]
    for index, item in enumerate(items):
        (Path(item["directory"])/"preparation").mkdir(parents=True,exist_ok=True)
    state.publish_skipped(items[0],{"case_id":items[0]["case_id"],"reason":"h3_timeline_ineligible"})
    for item in items[1:]:
        state.atomic_json(Path(item["directory"])/"generation"/"manifest.json",{})
    limits = {item["case_id"]:{"prepare":2,"generate":2} for item in items}
    counts = state.inventory(items,limits)
    assert (counts["done"],counts["skipped"]) == (2,1)
    assert state.finished(counts,3) is True


def test_supported_group_partitions_are_disjoint_and_complete():
    from r2v_data_v2.person_replacement.h3_pair_state import partition

    cases = list(range(19))
    for size in (2,4,8):
        parts = [partition(cases,worker,size) for worker in range(size)]
        assert sorted(item for part in parts for item in part) == cases
        for worker,part in enumerate(parts):
            assert all(item % size == worker for item in part)
