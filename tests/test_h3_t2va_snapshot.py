import json

import pytest

from r2v_data_v2.h3 import t2va_production as production
from tests.test_h3_t2va_production import FakeProcessor, sample_rows


def test_incomplete_shard_snapshot_and_later_completion(tmp_path):
    processor = FakeProcessor()
    processor.interrupt = ("clip327", "t2va")
    with pytest.raises(KeyboardInterrupt):
        production.process_shard(
            tmp_path, 0, sample_rows(330), processor, request_workers=1
        )
    shard = tmp_path / "shards" / production.shard_name(0)
    with (shard / "exports/t2va.jsonl.partial").open("ab") as handle:
        handle.write(b'{"broken":')
    snapshot = production.build_snapshot(tmp_path, "first")
    assert len(list(production.complete_rows(snapshot / "t2va.jsonl"))) == 327
    videos = list(production.complete_rows(snapshot / "videos.jsonl"))
    assert len(videos) == 327
    assert videos[0]["tasks"] == ["t2va", "ta2va_full_audio"]
    assert (tmp_path / "snapshots/LATEST").read_text() == "first\n"
    before = (snapshot / "t2va.jsonl").read_bytes()
    processor.interrupt = None
    production.process_shard(
        tmp_path, 0, sample_rows(330), processor, request_workers=1
    )
    second = production.build_snapshot(tmp_path, "second")
    assert len(list(production.complete_rows(second / "t2va.jsonl"))) == 330
    assert (snapshot / "t2va.jsonl").read_bytes() == before
    with pytest.raises(FileExistsError):
        production.build_snapshot(tmp_path, "first")


def test_snapshot_dedup_and_conflict(tmp_path):
    production.process_shard(tmp_path, 0, sample_rows(1), FakeProcessor())
    exports = tmp_path / "shards" / production.shard_name(0) / "exports"
    original = next(production.complete_rows(exports / "t2va.jsonl"))
    production.append_row(exports / "t2va.jsonl.partial", original)
    output = production.build_snapshot(tmp_path, "dedup")
    assert len(list(production.complete_rows(output / "t2va.jsonl"))) == 1
    production.append_row(
        exports / "t2va.jsonl.partial", {**original, "caption": "different"}
    )
    with pytest.raises(ValueError, match="conflict"):
        production.build_snapshot(tmp_path, "conflict")
    assert not (tmp_path / "snapshots/conflict").exists()
    assert (tmp_path / "snapshots/LATEST").read_text() == "dedup\n"


def test_t2va_ready_survives_ta_failure_in_snapshot(tmp_path):
    processor = FakeProcessor()
    processor.fail.add(("clip0", "ta2va"))
    production.process_shard(tmp_path, 0, sample_rows(1), processor)
    output = production.build_snapshot(tmp_path, "only-t2va")
    assert list(production.complete_rows(output / "videos.jsonl")) == [
        {"video": "/video/0.mp4", "tasks": ["t2va"]}
    ]
    assert json.loads((output / "summary.json").read_text())["counts"]["t2va"] == 1
