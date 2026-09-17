import json

import pytest

from r2v_data_v2.h3 import t2va_production as production


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
