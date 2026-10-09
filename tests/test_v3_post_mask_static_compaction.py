"""Rank-zero publication waits only on existing formal group markers."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from r2v_data_v2.v3 import config as config_module
from r2v_data_v2.v3 import post_mask_epoch_production as production
from r2v_data_v2.v3.post_mask_epoch_groups import group_root

REPO = Path(__file__).resolve().parents[1]


def _main_fixture(tmp_path, monkeypatch, *, count=48, rank=0, world_size=7):
    spec = importlib.util.spec_from_file_location(
        "static_compaction_tool", REPO / "tools/run_v3_post_mask_full_production.py"
    )
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    root = tmp_path / "in_pair_reference"
    stage2 = tmp_path / "entity_mask"
    (stage2 / "parts").mkdir(parents=True)
    for index in range(count):
        (stage2 / "parts" / f"shard-{index:09d}-{index:09d}.jsonl").write_text(
            "", encoding="utf-8"
        )
    config = config_module.load_config(tool.BASE_CONFIG)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", tmp_path / "private")
    monkeypatch.setattr(config_module, "OFFICIAL_POST_MASK_EXPORT_ROOT", root)
    monkeypatch.setattr(
        tool, "load_config", lambda _: replace(config, export_root=root / "shards")
    )
    events = []
    monkeypatch.setattr(tool, "_emit", lambda event, **details: events.append((event, details)))
    monkeypatch.setattr(tool, "write_source_descriptor", lambda *_: tmp_path / "source.yaml")
    compactions = []

    def compact(**kwargs):
        compactions.append(kwargs)
        return {"total_samples": count}

    monkeypatch.setattr(tool, "compact_production_exports", compact)
    argv = [
        "--entity-mask-root", str(stage2), "--group-size", "1",
        "--rank", str(rank), "--world-size", str(world_size),
    ]
    return tool, root, events, compactions, argv


def _complete(root, group):
    production.write_production_group_complete(
        root, group,
        {
            "completed": True,
            "subject_attributes_completed": True,
            "export_completed": True,
            "export_stats": {
                shard: {"sample_count": 1} for shard in group.canonical_shards
            },
        },
    )


@pytest.mark.parametrize("rank", [1, 6])
def test_nonzero_rank_exits_without_global_wait_or_compaction(tmp_path, monkeypatch, rank):
    tool, root, events, compactions, argv = _main_fixture(
        tmp_path, monkeypatch, count=1, rank=rank
    )
    monkeypatch.setattr(tool, "run_elastic_groups", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tool, "production_group_completed", lambda *_: pytest.fail("checked remote markers")
    )
    monkeypatch.setattr(
        tool, "compact_if_complete", lambda *_: pytest.fail("nonzero rank compacted")
    )
    assert tool.main(argv) == 0
    assert compactions == []
    assert not (root / "state/compaction_completed.json").exists()
    assert "post_mask_production_complete" not in [event for event, _ in events]


def test_rank0_waits_for_last_remote_g34_using_only_small_markers(tmp_path, monkeypatch):
    tool, root, events, compactions, argv = _main_fixture(tmp_path, monkeypatch)
    pending = []
    original_read = Path.read_text

    def finish_local(state, groups, _runner, **_kwargs):
        for group in groups:
            if group.group_index == 34:
                pending.append(group)
            else:
                _complete(state, group)
        def read_marker(path, *args, **kwargs):
            assert path.name == "completed.json", "read receipts or media while waiting"
            return original_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_marker)
        for method in ("glob", "rglob", "iterdir", "read_bytes"):
            monkeypatch.setattr(
                Path, method, lambda *_args, **_kwargs: pytest.fail("scanned or hashed tree")
            )
        monkeypatch.setattr(type(groups[0]), "identity", lambda *_: pytest.fail("hashed group"))

    monkeypatch.setattr(tool, "run_elastic_groups", finish_local)
    sleeps = []

    def remote_finishes(seconds):
        assert compactions == []
        assert not (root / "state/compaction_completed.json").exists()
        assert "post_mask_production_complete" not in [event for event, _ in events]
        sleeps.append(seconds)
        if len(sleeps) == 2:
            _complete(root / "state", pending[0])

    monkeypatch.setattr(time, "sleep", remote_finishes)
    assert tool.main(argv) == 0
    assert len(sleeps) == 2 and all(seconds > 0 for seconds in sleeps)
    assert len(compactions) == 1
    assert json.loads(original_read(root / "state/compaction_completed.json")) == {
        "status": "completed", "group_count": 48,
    }
    waiting = [
        details["remaining_group_count"] for event, details in events
        if event == "post_mask_production_waiting_for_global_completion"
    ]
    assert waiting == [1, 1]
    assert [event for event, _ in events].count("post_mask_production_complete") == 1


def test_rank0_does_not_report_complete_while_remote_marker_is_missing(tmp_path, monkeypatch):
    tool, root, events, compactions, argv = _main_fixture(tmp_path, monkeypatch, count=1)
    monkeypatch.setattr(tool, "run_elastic_groups", lambda *_args, **_kwargs: None)

    def stop_wait(_seconds):
        raise InterruptedError("test stopped waiting")

    monkeypatch.setattr(time, "sleep", stop_wait)
    with pytest.raises(InterruptedError, match="test stopped waiting"):
        tool.main(argv)
    assert compactions == []
    assert not (root / "state/compaction_completed.json").exists()
    assert "post_mask_production_complete" not in [event for event, _ in events]


def test_single_node_compacts_immediately_after_its_groups_finish(tmp_path, monkeypatch):
    tool, root, events, compactions, argv = _main_fixture(
        tmp_path, monkeypatch, count=1, world_size=1
    )

    def finish(state, groups, _runner, **_kwargs):
        _complete(state, groups[0])

    monkeypatch.setattr(tool, "run_elastic_groups", finish)
    monkeypatch.setattr(time, "sleep", lambda *_: pytest.fail("completed groups slept"))
    assert tool.main(argv) == 0
    assert len(compactions) == 1
    assert (root / "state/compaction_completed.json").is_file()
    assert "post_mask_production_complete" in [event for event, _ in events]


@pytest.mark.parametrize("marker", ["{broken", '{"status":"completed"}'])
def test_corrupt_remote_marker_fails_even_with_an_earlier_missing_marker(
    tmp_path, monkeypatch, marker
):
    tool, root, events, compactions, argv = _main_fixture(tmp_path, monkeypatch)

    def finish(state, groups, _runner, **_kwargs):
        for group in groups[1:]:
            if group.group_index != 34:
                _complete(state, group)
        corrupt = group_root(state, groups[34]) / "completed.json"
        corrupt.parent.mkdir(parents=True, exist_ok=True)
        corrupt.write_text(marker, encoding="utf-8")

    monkeypatch.setattr(tool, "run_elastic_groups", finish)
    monkeypatch.setattr(time, "sleep", lambda *_: pytest.fail("corrupt marker waited"))
    with pytest.raises(ValueError, match="production group marker"):
        tool.main(argv)
    assert compactions == []
    assert not (root / "state/compaction_completed.json").exists()
    assert "post_mask_production_complete" not in [event for event, _ in events]


def test_runner_error_propagates_without_waiting_or_compacting(tmp_path, monkeypatch):
    tool, root, events, compactions, argv = _main_fixture(
        tmp_path, monkeypatch, count=1, world_size=1
    )

    def fail_runner(*_args):
        raise RuntimeError("runner failed")

    monkeypatch.setattr(tool, "run_removal_epoch", fail_runner)
    monkeypatch.setattr(time, "sleep", lambda *_: pytest.fail("runner error waited"))
    with pytest.raises(RuntimeError, match="runner failed"):
        tool.main(argv)
    assert compactions == []
    assert not (root / "state/compaction_completed.json").exists()
    assert "post_mask_production_complete" not in [event for event, _ in events]
