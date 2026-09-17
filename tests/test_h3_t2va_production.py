import json
from collections import Counter

import pytest

from r2v_data_v2.h3 import t2va_production as production
from tests import test_h3_t2va_shadow as shared

finalized = shared.finalized
setup = shared.setup
ffmpeg = shared.ffmpeg


def test_shard_59_and_schedule():
    assert production.shard_bounds(59) == (590000, 599999)
    assert production.shard_name(59) == "shard-000590000-000599999"
    schedules = [
        production.shard_order(start=0, end=379, seed=20260917, rank=r, world_size=4)
        for r in range(4)
    ]
    assert len(set().union(*map(set, schedules))) == 380
    assert sum(map(len, schedules)) == 380
    assert schedules[0] == production.shard_order(
        start=0, end=379, seed=20260917, rank=0, world_size=4
    )
    assert production.shard_order(explicit=[59, 12, 87], rank=99, world_size=1) == [
        59,
        12,
        87,
    ]


def test_index_and_seek_preserve_raw_rows(tmp_path):
    source = tmp_path / "source.jsonl"
    with source.open("wb") as handle:
        for i in range(20003):
            handle.write((json.dumps({"index": i}) + "\n").encode())
    root = tmp_path / "output"
    index = production.build_source_index(source, root)
    assert index["source_record_count"] == 20003
    assert len(index["shards"]) == 3
    shard = production.materialize_shard(index, 1, root)
    rows = [json.loads(line) for line in shard.read_text().splitlines()]
    assert [r["index"] for r in rows] == list(range(10000, 20000))
    assert production.build_source_index(source, root) == index
    source.write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match="source"):
        production.build_source_index(source, root)


def test_invalid_rows_do_not_shift_shards(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"not json\n" + b"{}\n" * 9999 + b'{"last":true}\n')
    root = tmp_path / "output"
    index = production.build_source_index(source, root)
    assert (
        production.materialize_shard(index, 1, root).read_bytes() == b'{"last":true}\n'
    )


class FakeProcessor:
    def __init__(self):
        self.calls = Counter()
        self.fail = set()
        self.interrupt = None

    def identity(self, row):
        return str(row["clip_uid"])

    def process(self, stage, row, temporary, destination):
        key = (row["clip_uid"], stage)
        self.calls[key] += 1
        if key == self.interrupt:
            raise KeyboardInterrupt()
        if key in self.fail:
            raise ValueError("synthetic sample failure")
        (temporary / "result.txt").write_text(str(key))
        return {"exports": {}, "model_call_count": 1}


def sample_rows(n=3):
    return [
        {"source_index": i, "clip_uid": f"clip{i}", "video": f"/video/{i}.mp4"}
        for i in range(n)
    ]


def test_failure_isolation_and_stage_resume(tmp_path):
    processor = FakeProcessor()
    processor.fail = {("clip1", "t2va"), ("clip2", "ta2va")}
    production.process_shard(tmp_path, 0, sample_rows(), processor, request_workers=1)
    states = production.load_states(tmp_path / "shards" / production.shard_name(0))
    assert states["clip0"]["ta2va_status"] == "ready"
    assert states["clip1"]["t2va_status"] == "failed"
    assert states["clip1"]["ta2va_status"] == "skipped"
    assert states["clip2"]["t2va_status"] == "ready"
    assert states["clip2"]["ta2va_status"] == "failed"
    processor.fail.clear()
    production.process_shard(tmp_path, 0, sample_rows(), processor, request_workers=1)
    assert processor.calls["clip0", "t2va"] == 1
    assert processor.calls["clip1", "t2va"] == 2
    assert processor.calls["clip2", "t2va"] == 1
    assert processor.calls["clip2", "ta2va"] == 2


def test_interrupt_and_artifact_recovery(tmp_path):
    processor = FakeProcessor()
    processor.interrupt = ("clip3", "t2va")
    with pytest.raises(KeyboardInterrupt):
        production.process_shard(
            tmp_path, 0, sample_rows(5), processor, request_workers=1
        )
    shard = tmp_path / "shards" / production.shard_name(0)
    (shard / "state.jsonl.partial").write_bytes(b'{"partial":')
    processor.interrupt = None
    production.process_shard(tmp_path, 0, sample_rows(5), processor, request_workers=1)
    assert all(processor.calls[f"clip{i}", "t2va"] == 1 for i in range(3))
    assert processor.calls["clip4", "ta2va"] == 1


def test_skipped_and_lock(tmp_path):
    processor = FakeProcessor()
    rows = sample_rows(1)
    rows[0]["upstream_failure"] = "audio_preprocessing_required"
    production.process_shard(tmp_path, 0, rows, processor)
    production.process_shard(tmp_path, 0, rows, processor)
    assert not processor.calls
    with (
        production.file_lock(tmp_path / "lock"),
        pytest.raises(production.ShardLockedError),
        production.file_lock(tmp_path / "lock"),
    ):
        pass


def test_endpoint_outage_stops_scheduling(tmp_path):
    class Offline(FakeProcessor):
        def process(self, *args):
            raise production.EndpointUnavailable("connection refused")

    with pytest.raises(production.EndpointUnavailable):
        production.process_shard(
            tmp_path, 0, sample_rows(100), Offline(), request_workers=1
        )
    assert not production.load_states(tmp_path / "shards" / production.shard_name(0))


def test_t2va_shadow_equivalence(tmp_path, finalized):
    from tests import test_h3_t2va_shadow as shared

    inventory = shared.build(finalized, tmp_path)
    drafts = [
        shared.draft_for(j).model_dump_json()
        for j in inventory.jobs
        if not j.upstream_failure
    ]
    old_backend = shared.T2VAMimoBackend(
        shared.config(tmp_path), client=shared.Client(drafts)
    )
    shared.t2va.run_t2va_shadow(inventory, old_backend)
    old_root = shared.t2va.t2va_root(finalized[0], "t2va-test")
    new_backend = shared.T2VAMimoBackend(
        shared.config(tmp_path), client=shared.Client(drafts)
    )
    for job in inventory.jobs:
        if job.upstream_failure:
            continue
        temporary = tmp_path / "new" / job.clip_uid
        temporary.mkdir(parents=True)
        result = production.t2va_stage(job, new_backend, temporary)
        assert json.loads((temporary / "core.json").read_text()) == json.loads(
            (old_root / "core" / f"{job.clip_uid}.json").read_text()
        )
        assert (
            result["exports"]["t2va"]["caption"]
            == (old_root / "prompts" / f"{job.clip_uid}.txt").read_text()
        )
        assert result["model_call_count"] == 2
