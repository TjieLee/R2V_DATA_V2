"""Execution-only priority probes against isolated synthetic state."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from r2v_data_v2.v3 import post_mask_epoch_production as production
from r2v_data_v2.v3.post_mask_epoch_groups import (
    build_groups,
    group_root,
    write_group_descriptor,
)


def _groups(count):
    return build_groups([f"shard-{i}" for i in range(count)], campaign={}, group_size=1)


def _marker(root, group, stage, payload=None):
    path = group_root(root, group) / "composition" / f"{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {}))


def _run(root, groups, **options):
    visited = []

    def runner(group, ledger, emit):
        visited.append(group.group_id)
        return {
            "completed": True,
            "subject_attributes_completed": True,
            "export_completed": True,
            "export_stats": {s: {"sample_count": 0} for s in group.canonical_shards},
        }

    production.run_elastic_groups(root, groups, runner, sleep=lambda _: None, **options)
    return visited


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_all_ranks_preserve_stage_priority(tmp_path, rank, monkeypatch):
    groups = _groups(8)
    stages = [
        None,
        "pair_started",
        "reference_edit_started",
        "reference_integrity_started",
        "instruct_started",
        "subject_attributes_started",
        "subject_attributes_completed",
    ]
    for group, stage in zip(groups, stages, strict=False):
        if stage:
            _marker(tmp_path, group, stage)
    # Already completed groups never enter priority probes or their runner.
    production.write_production_group_complete(
        tmp_path,
        groups[7],
        {
            "completed": True,
            "subject_attributes_completed": True,
            "export_completed": True,
            "export_stats": {"shard-7": {"sample_count": 0}},
        },
    )
    monkeypatch.setattr(
        "os.scandir", lambda *_: pytest.fail("scheduler scanned outcomes")
    )
    assert _run(tmp_path, groups, rank=rank, world_size=3) == [
        g.group_id for g in reversed(groups[:7])
    ]


def test_equal_priority_keeps_rank_rotation(tmp_path):
    groups = _groups(4)
    assert _run(tmp_path, groups, rank=1, world_size=2) == [
        g.group_id for g in groups[2:] + groups[:2]
    ]


def test_claimed_group_precedes_new_group_with_missing_hints(tmp_path):
    groups = _groups(2)
    write_group_descriptor(tmp_path, groups[1])
    assert _run(
        tmp_path, groups, rank=0, world_size=1, priority_hints=tmp_path / "missing.json"
    ) == [
        groups[1].group_id,
        groups[0].group_id,
    ]


def test_hints_sort_sa_progress_only_and_ignore_unknown_invalid_entries(tmp_path):
    groups = _groups(5)
    for group in groups[:3]:
        _marker(tmp_path, group, "subject_attributes_started")
    _marker(tmp_path, groups[3], "subject_attributes_completed")
    hints = tmp_path / "hints.json"
    hints.write_text(
        json.dumps(
            {
                groups[0].group_id: {"terminal": 10, "eligible": 100},
                groups[1].group_id: {"terminal": 99, "eligible": 100},
                groups[2].group_id: {"terminal": -1, "eligible": 100},
                groups[4].group_id: {"terminal": 100, "eligible": 100},
                "unknown": {"terminal": 100, "eligible": 100},
            }
        )
    )
    assert _run(tmp_path, groups, rank=1, world_size=2, priority_hints=hints) == [
        groups[i].group_id for i in (3, 1, 0, 2, 4)
    ]


@pytest.mark.parametrize("contents", ["invalid JSON", "[]", "null"])
def test_malformed_hints_fall_back_to_markers(tmp_path, contents):
    groups = _groups(2)
    _marker(tmp_path, groups[1], "subject_attributes_started")
    hints = tmp_path / "hints.json"
    hints.write_text(contents)
    assert _run(tmp_path, groups, rank=0, world_size=1, priority_hints=hints) == [
        groups[1].group_id,
        groups[0].group_id,
    ]


def test_one_shot_tool_counts_only_eligible_outcome_files(tmp_path, monkeypatch):
    from tools import create_v3_post_mask_priority_hints as tool

    groups = _groups(2)
    group = groups[0]
    state = tmp_path / "state"
    root = group_root(state, group)
    _marker(
        state,
        group,
        "subject_attributes_started",
        {
            "eligible_clip_uids_by_shard": {"shard-0": ["a", "b", "c"]},
        },
    )
    outcomes = root / "semantic/subject_attributes/outcomes/shard-0"
    outcomes.mkdir(parents=True)
    for name in ("a.json", "unrelated.json", ".b.json.tmp"):
        (outcomes / name).write_text("intentionally not JSON")
    (outcomes / "b.json").symlink_to(outcomes / "a.json")
    (outcomes / "c.json").mkdir()
    real_read = Path.read_text
    real_scandir = __import__("os").scandir
    scanned = []

    def guarded_scan(path):
        scanned.append(Path(path))
        return real_scandir(path)

    def guarded_read(path, *args, **kwargs):
        assert "outcomes" not in path.parts, "tool opened outcome contents"
        return real_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    monkeypatch.setattr("os.scandir", guarded_scan)
    output = tmp_path / "private/hints.json"
    assert tool.main(["--state-root", str(state), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == {
        group.group_id: {"terminal": 1, "eligible": 3},
    }
    assert scanned == [state / "resource_epochs", outcomes]


def test_formal_launcher_exposes_optional_hints_cli():
    from tools.run_v3_post_mask_full_production import _parser

    assert _parser().parse_args(
        ["--priority-hints", "/tmp/hints.json"]
    ).priority_hints == Path("/tmp/hints.json")


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_formal_shell_forwards_hints_to_elastic_consumer(tmp_path, rank):
    """Dropping shell "$@" or CLI priority_hints forwarding breaks this test."""
    repo = Path(__file__).resolve().parents[1]
    sandbox = tmp_path / "checkout"
    for directory in ("scripts", "tools", "configs", ".venv/bin"):
        (sandbox / directory).mkdir(parents=True)
    script = sandbox / "scripts/run_v3_post_mask_full_production.sh"
    # Execute the real shell unchanged, with only its runtime/environment faked.
    shutil.copyfile(repo / "scripts" / script.name, script)
    python = sandbox / ".venv/bin/python"
    # Execute the original venv path, rather than relocating its binary symlink.
    python.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    (sandbox / ".venv/bin/activate").write_text(":\n")
    official = tmp_path / "official"
    (sandbox / "server_env.sh").write_text(f"production='{official}'\n")
    config = yaml.safe_load(
        (repo / "configs/v3_post_mask_resource_epoch_production.yaml").read_text()
    )
    config.update(run_root=str(tmp_path / "run"), export_root=str(official / "shards"))
    (sandbox / "configs/v3_post_mask_resource_epoch_production.yaml").write_text(
        yaml.safe_dump(config)
    )
    fixture = tmp_path / "fixture"
    (fixture / "parts").mkdir(parents=True)
    (fixture / "parts/shard-000000000-000000029.jsonl").write_text("")
    hints = tmp_path / "private hints.json"
    hints.write_text("{}")
    observed = tmp_path / "consumer.json"
    (sandbox / "tools/run_v3_post_mask_full_production.py").write_text(
        f"""import json
import sys
from pathlib import Path
sys.path.insert(0, {str(repo)!r})
from tools import run_v3_post_mask_full_production as cli
# Keep the real CLI/config/group builder; isolate physical destinations only.
cli.config_module.ALLOWED_WRITABLE_ROOT = Path({str(tmp_path)!r})
cli.config_module.OFFICIAL_POST_MASK_EXPORT_ROOT = Path({str(official)!r})
def consume(root, groups, runner, *, rank, world_size, emit, priority_hints):
    Path({str(observed)!r}).write_text(json.dumps({{
        "priority_hints": str(priority_hints), "rank": rank,
        "world_size": world_size, "state_root": str(root),
        "group_count": len(groups),
    }}))
    # Stop before any real runner, model, compaction or durable state operation.
    raise SystemExit(0)
cli.run_elastic_groups = consume
raise SystemExit(cli.main())
"""
    )
    result = subprocess.run(
        ["bash", str(script), "--entity-mask-root", str(fixture),
         "--priority-hints", str(hints)],
        env={**os.environ, "RANK": str(rank), "WORLD_SIZE": "3"},
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(observed.read_text()) == {
        "priority_hints": str(hints), "rank": rank, "world_size": 3,
        "state_root": str(official / "state"), "group_count": 1,
    }
    assert not (official / "state").exists()
