"""Synthetic checks for elastic formal Post-Mask group ownership."""

from __future__ import annotations

import importlib.util
import os
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.v3 import config as config_module
from r2v_data_v2.v3 import post_mask_epoch_production as production
from r2v_data_v2.v3 import post_mask_epoch_removal as removal
from r2v_data_v2.v3.post_mask_epoch_groups import build_groups, group_ownership
from r2v_data_v2.v3.post_mask_production import ShardPaths

REPO = Path(__file__).resolve().parents[1]


def _groups(count=4):
    return build_groups(
        [f"shard-{i:09d}-{i:09d}" for i in range(count)],
        campaign={"source": "synthetic"},
        group_size=1,
    )


def _completed(group, _ledger, _emit):
    return {
        "completed": True,
        "subject_attributes_completed": True,
        "export_completed": True,
        "export_stats": {shard: {"sample_count": 1} for shard in group.canonical_shards},
    }


def test_elastic_restart_with_fewer_nodes_claims_every_remaining_group(tmp_path):
    groups = _groups(5)
    visited = []

    def interrupted(group, ledger, emit):
        if len(visited) == 2:
            raise RuntimeError("node lost")
        visited.append(group.group_id)
        return _completed(group, ledger, emit)

    with pytest.raises(RuntimeError, match="node lost"):
        production.run_elastic_groups(
            tmp_path, groups, interrupted, rank=3, world_size=4, sleep=lambda _: None
        )
    production.run_elastic_groups(
        tmp_path, groups, _completed, rank=0, world_size=1, sleep=lambda _: None
    )
    assert all(production.production_group_completed(tmp_path, group) for group in groups)
    assert len(visited) == 2


def test_busy_group_is_rescanned_not_abandoned(tmp_path, monkeypatch):
    group = _groups(1)[0]
    original = production.group_ownership
    attempts = []
    sleeps = []

    @contextmanager
    def busy_once(root, candidate):
        attempts.append(candidate.group_id)
        if len(attempts) == 1:
            yield False
        else:
            with original(root, candidate) as held:
                yield held

    monkeypatch.setattr(production, "group_ownership", busy_once)
    production.run_elastic_groups(
        tmp_path, (group,), _completed,
        rank=0, world_size=2, sleep=sleeps.append,
    )
    assert attempts == [group.group_id, group.group_id]
    assert len(sleeps) == 1
    assert production.production_group_completed(tmp_path, group)


def test_group_flock_excludes_second_writer(tmp_path):
    group = _groups(1)[0]
    with (
        group_ownership(tmp_path, group) as first,
        group_ownership(tmp_path, group) as second,
    ):
        assert first is True
        assert second is False


def test_completed_fast_skip_never_validates_hash_or_calls_runner(
    tmp_path, monkeypatch
):
    group = _groups(1)[0]
    production.write_production_group_complete(tmp_path, group, _completed(group, None, None))
    monkeypatch.setattr(
        type(group), "identity", lambda self: pytest.fail("completion calculated SHA"),
    )
    production.run_elastic_groups(
        tmp_path, (group,),
        lambda *_: pytest.fail("completed group entered runner"),
        rank=0, world_size=1, sleep=lambda _: None,
    )


def test_incomplete_outcome_remains_claimable_on_next_invocation(tmp_path):
    group = _groups(1)[0]
    with pytest.raises(production.IncompleteGroupError):
        production.run_elastic_groups(
            tmp_path, (group,),
            lambda *_: {"completed": False, "reason": "request unavailable"},
            rank=0, world_size=1, sleep=lambda _: None,
        )
    assert not production.production_group_completed(tmp_path, group)
    production.run_elastic_groups(
        tmp_path, (group,), _completed,
        rank=0, world_size=2, sleep=lambda _: None,
    )
    assert production.production_group_completed(tmp_path, group)


def test_incomplete_group_does_not_block_later_groups(tmp_path):
    groups = _groups(3)
    visited = []

    def runner(group, ledger, emit):
        visited.append(group.group_id)
        if group == groups[0]:
            return {"completed": False, "reason": "retryable request failure"}
        return _completed(group, ledger, emit)

    with pytest.raises(production.IncompleteGroupError):
        production.run_elastic_groups(
            tmp_path, groups, runner, rank=0, world_size=1, sleep=lambda _: None
        )
    assert visited == [group.group_id for group in groups]
    assert not production.production_group_completed(tmp_path, groups[0])
    assert all(production.production_group_completed(tmp_path, group) for group in groups[1:])


def test_formal_shard_paths_write_exports_under_official_root(tmp_path, monkeypatch):
    writable = tmp_path / "private"
    official = tmp_path / "public" / "in_pair_reference"
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(config_module, "OFFICIAL_POST_MASK_EXPORT_ROOT", official)
    shard = "shard-000000000-000009999"
    paths = ShardPaths.for_shard(
        official / "state",
        tmp_path / "entity_mask" / "parts" / f"{shard}.jsonl",
        runs_root=writable / "r2v_v3_runs/production/jea_motion_v1/in_pair_reference",
        exports_root=official / "shards",
    )
    assert paths.export_root == official / "shards" / shard
    assert paths.state_root == official / "state" / "shards" / shard
    assert paths.run_root.is_relative_to(writable)


def test_formal_export_completed_fast_skip_does_not_rehash(tmp_path, monkeypatch):
    marker = tmp_path / "state" / "shard-000000000-000009999" / "completed.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(
        '{"status":"completed","shard_id":"shard-000000000-000009999",'
        '"eligible_rows":2,"terminal_rows":2,"export_completed":true,'
        '"sample_count":1}',
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        state_root=marker.parent,
        export_root=tmp_path / "exports" / "shard-000000000-000009999",
    )
    monkeypatch.setattr(
        removal, "expected_shard_completion", lambda *_: pytest.fail("tree hashed")
    )
    result = removal.export_shard(
        SimpleNamespace(root=tmp_path / "run"), paths, ("a", "b"), formal_production=True
    )
    assert result == {"sample_count": 1, "rebuilt": False}


def test_formal_export_missing_marker_uses_existing_exporter(tmp_path, monkeypatch):
    paths = SimpleNamespace(
        state_root=tmp_path / "state" / "shard-000000000-000009999",
        export_root=tmp_path / "exports" / "shard-000000000-000009999",
    )
    calls = []

    class FakeExporter:
        def __init__(self, config, storage):
            calls.append("construct")

        def export(self, *, overwrite):
            calls.append(("export", overwrite))
            return SimpleNamespace(sample_count=2)

    monkeypatch.setattr(removal, "expected_shard_completion", lambda *_: pytest.fail("tree hashed"))
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_runtime._ShardStorage",
        lambda storage, clips: SimpleNamespace(config=storage.config),
    )
    monkeypatch.setattr(
        "r2v_data_v2.v3.subject_attributes.reconcile_subject_attribute_outputs",
        lambda **_: calls.append("reconcile"),
    )
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_runtime._cleanup_export_staging",
        lambda *_: calls.append("cleanup"),
    )
    monkeypatch.setattr(
        "r2v_data_v2.v3.post_mask_runtime._export_identity",
        lambda *_: {"semantic": "test"},
    )
    monkeypatch.setattr("r2v_data_v2.v3.storage.DatasetExporter", FakeExporter)
    storage = SimpleNamespace(root=tmp_path / "run", config=object())
    result = removal.export_shard(
        storage, paths, ("a", "b"), formal_production=True
    )
    assert result == {"sample_count": 2, "rebuilt": True}
    assert calls == ["reconcile", "cleanup", "construct", ("export", False)]
    assert (paths.state_root / "export_identity.json").is_file()
    assert production.production_shard_completed(paths, ("a", "b")) == 2


def test_formal_export_cleans_only_owned_stale_backup(tmp_path):
    shard = "shard-000000000-000009999"
    destination = tmp_path / "shards" / shard
    destination.parent.mkdir()
    owned = destination.with_name(f".{shard}.backup-" + "a" * 32)
    owned.mkdir()
    (owned / "stale").write_text("old", encoding="utf-8")
    unrelated = destination.with_name(f".{shard}.backup-not-a-uuid")
    unrelated.mkdir()
    production.cleanup_export_backups(SimpleNamespace(export_root=destination))
    assert not owned.exists()
    assert unrelated.is_dir()


def test_committed_config_pins_validated_models_and_official_output():
    from r2v_data_v2.v3.config import BOOGU_REMOVE_BACKEND, load_config

    config = load_config(REPO / "configs/v3_post_mask_resource_epoch_production.yaml")
    qwen = "/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct"
    assert {service.model for _, service in config.qwen_services()} == {qwen}
    assert len(config.qwen_services()) == 8
    assert config.remove.backend == BOOGU_REMOVE_BACKEND
    assert config.reference_edit.enabled
    assert config.reference_integrity.enabled
    assert config.export_root == config_module.OFFICIAL_POST_MASK_EXPORT_ROOT / "shards"


def test_one_command_dry_run_needs_no_post_mask_environment_or_models():
    env = {k: v for k, v in os.environ.items() if not k.startswith("POST_MASK_")}
    env.update(RANK="2", WORLD_SIZE="4")
    result = subprocess.run(
        ["bash", str(REPO / "scripts/run_v3_post_mask_full_production.sh"), "--dry-run"],
        env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "rank=2 world_size=4" in result.stdout
    assert "entity_mask" in result.stdout
    assert "in_pair_reference" in result.stdout
    assert "Qwen3-VL-8B-Instruct" in result.stdout
    assert "v3_post_mask_resource_epoch_production.yaml" in result.stdout
    assert "run_removal_epoch" not in result.stderr


def test_one_command_dry_run_defaults_to_single_node():
    env = {
        k: v for k, v in os.environ.items()
        if not k.startswith("POST_MASK_") and k not in {"RANK", "WORLD_SIZE"}
    }
    result = subprocess.run(
        ["bash", str(REPO / "scripts/run_v3_post_mask_full_production.sh"), "--dry-run"],
        env=env, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0
    assert "rank=0 world_size=1" in result.stdout


def test_compactor_accepts_only_exact_official_export_layout(tmp_path, monkeypatch):
    from tools.compact_v3_production_exports import _validate_roots

    official = tmp_path / "public" / "in_pair_reference"
    (official / "shards").mkdir(parents=True)
    monkeypatch.setattr(config_module, "OFFICIAL_POST_MASK_EXPORT_ROOT", official)
    assert _validate_roots(official / "shards", official) == (
        official / "shards", official
    )
    with pytest.raises(ValueError):
        _validate_roots(official / "shards", tmp_path / "other")


def _load_full_tool():
    path = REPO / "tools/run_v3_post_mask_full_production.py"
    spec = importlib.util.spec_from_file_location("full_post_mask_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_tool_publishes_one_official_root_and_pins_qwen(tmp_path, monkeypatch):
    tool = _load_full_tool()
    official = tmp_path / "public" / "in_pair_reference"
    stage2 = tmp_path / "entity_mask"
    (stage2 / "parts").mkdir(parents=True)
    (stage2 / "parts/shard-000000000-000009999.jsonl").write_text("", encoding="utf-8")
    config = config_module.load_config(
        REPO / "configs/v3_post_mask_resource_epoch_production.yaml"
    )
    monkeypatch.setattr(config_module, "OFFICIAL_POST_MASK_EXPORT_ROOT", official)
    monkeypatch.setattr(
        tool, "load_config", lambda _: replace(config, export_root=official / "shards")
    )
    seen = {}
    monkeypatch.setattr(tool, "run_elastic_groups", lambda root, groups, runner, **kw: seen.update(root=root, groups=groups, runner=runner, options=kw))
    monkeypatch.setattr(tool, "compact_if_complete", lambda root, groups, config, path: seen.update(compacted=root))
    monkeypatch.setenv("POST_MASK_QWEN_MODEL_PATH", "/stale/32B")
    monkeypatch.setenv("POST_MASK_FORMAL_PRODUCTION", "")
    monkeypatch.setenv("POST_MASK_ROOT", "")
    assert tool.main([
        "--base-config", str(REPO / "configs/v3_post_mask_resource_epoch_production.yaml"),
        "--entity-mask-root", str(stage2), "--rank", "1", "--world-size", "3",
    ]) == 0
    assert seen["root"] == official / "state"
    assert seen["compacted"] == official
    assert seen["options"]["rank"] == 1
    assert seen["options"]["world_size"] == 3
    assert os.environ["POST_MASK_QWEN_MODEL_PATH"].endswith("Qwen3-VL-8B-Instruct")
    assert os.environ["POST_MASK_FORMAL_PRODUCTION"] == "1"
    assert os.environ["POST_MASK_ROOT"] == str(official / "state")


def test_completed_compaction_skips_source_and_media_hashes(tmp_path, monkeypatch):
    tool = _load_full_tool()
    root = tmp_path / "in_pair_reference"
    groups = _groups(1)
    production.write_production_group_complete(root / "state", groups[0], _completed(groups[0], None, None))
    marker = root / "state" / "compaction_completed.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('{"status":"completed","group_count":1}', encoding="utf-8")
    monkeypatch.setattr(tool, "write_source_descriptor", lambda *_: pytest.fail("source hashed"))
    monkeypatch.setattr(tool, "compact_production_exports", lambda **_: pytest.fail("tree hashed"))
    tool.compact_if_complete(root, groups, object(), Path("/unused/config.yaml"))
