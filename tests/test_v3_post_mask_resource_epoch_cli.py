"""CLI-level tests for the opt-in resource-epoch launcher.

These drive ``tools/run_v3_post_mask_resource_epoch.main`` itself, not a
helper, because the two bugs being guarded here are wiring bugs:

* a ``--dry-run`` that still executed the job runner;
* a launcher whose group identity came from parsed CLI values while an
  external job runner resolved its roots from the environment.

The job runner is replaced by a recorder that reads the same environment the
real ``run_removal_epoch`` reads, so "the runner saw X" is observable.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import pytest

import r2v_data_v2.v3.config as config_module

SHARD = "shard-000000000-000000000"


def _load_launcher() -> Any:
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_v3_post_mask_resource_epoch.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_v3_post_mask_resource_epoch", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeConfig:
    def __init__(self, fingerprint: str, dataset_json: str) -> None:
        self._fingerprint = fingerprint
        self.dataset_json = Path(dataset_json)

    def fingerprint(self) -> str:
        return self._fingerprint


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    module = _load_launcher()
    writable = tmp_path / "workspace" / "data"
    writable.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config_module, "ALLOWED_WRITABLE_ROOT", writable)
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", tmp_path / "public")
    monkeypatch.setattr(
        config_module, "ALLOWED_PRETRAINED_ROOT", tmp_path / "public" / "pretrained"
    )
    monkeypatch.setattr(
        config_module, "ALLOWED_USER_MODEL_ROOT", writable / "models"
    )
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda path: _FakeConfig(f"fp-{Path(path).name}", str(path)),
    )
    return module


def _stage2_root(tmp_path: Path, name: str = "entity_mask", shards: int = 1) -> Path:
    root = tmp_path / name
    parts = root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    for index in range(shards):
        (parts / f"shard-{index:09d}-{index:09d}.jsonl").write_text("")
    return root


def _install_recorder(
    monkeypatch: pytest.MonkeyPatch, launcher: Any, *, completed: bool = False
) -> dict[str, Any]:
    """Replace the job runner loader with one that records what it observes."""
    seen: dict[str, Any] = {"loads": 0, "calls": 0, "order": [], "completed": completed}

    def fake_load(spec: str) -> Any:
        seen["loads"] += 1
        seen["spec"] = spec

        def runner(group: Any, ledger: Any, emit: Any) -> dict[str, Any]:
            seen["calls"] += 1
            seen["group_id"] = getattr(group, "group_id", "")
            seen["order"].append(getattr(group, "group_id", ""))
            seen["ledger_root"] = Path(ledger.root)
            seen["env"] = {
                name: os.environ.get(name)
                for name in (
                    "POST_MASK_BASE_CONFIG",
                    "POST_MASK_TAG",
                    "POST_MASK_ENTITY_MASK_ROOT",
                    "POST_MASK_ROOT",
                    "POST_MASK_JOB_RUNNER",
                    "POST_MASK_CANONICAL_SHARD_COUNT",
                )
            }
            if seen["completed"]:
                return {"completed": True, "reason": ""}
            return {"completed": False, "reason": "recorder"}

        return runner

    monkeypatch.setattr(launcher, "_load_runner", fake_load)
    return seen


def test_dry_run_and_job_runner_are_mutually_exclusive(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _install_recorder(monkeypatch, launcher)
    root = _stage2_root(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        launcher.main(
            [
                "--base-config",
                str(tmp_path / "cfg.yaml"),
                "--entity-mask-root",
                str(root),
                "--dry-run",
                "--job-runner",
                "pkg.module:callable",
            ]
        )

    assert excinfo.value.code == 2
    # The parser rejects the combination before anything can be imported.
    assert seen["loads"] == 0
    assert seen["calls"] == 0


def test_dry_run_only_plans_and_never_loads_a_runner(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _install_recorder(monkeypatch, launcher)
    root = _stage2_root(tmp_path)

    assert (
        launcher.main(
            [
                "--base-config",
                str(tmp_path / "cfg.yaml"),
                "--entity-mask-root",
                str(root),
                "--dry-run",
            ]
        )
        == 0
    )
    assert seen["loads"] == 0
    assert seen["calls"] == 0


def test_neither_dry_run_nor_job_runner_fails_clearly(
    launcher: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_recorder(monkeypatch, launcher)
    root = _stage2_root(tmp_path)

    assert (
        launcher.main(
            [
                "--base-config",
                str(tmp_path / "cfg.yaml"),
                "--entity-mask-root",
                str(root),
            ]
        )
        == 2
    )
    assert "needs --job-runner" in capsys.readouterr().out


def test_cli_base_config_overrides_the_environment(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _install_recorder(monkeypatch, launcher)
    root = _stage2_root(tmp_path)
    monkeypatch.setenv("POST_MASK_BASE_CONFIG", str(tmp_path / "env.yaml"))
    monkeypatch.setenv("POST_MASK_ENTITY_MASK_ROOT", str(root))

    launcher.main(
        [
            "--base-config",
            str(tmp_path / "cli.yaml"),
            "--entity-mask-root",
            str(root),
            "--job-runner",
            "pkg.module:callable",
        ]
    )

    assert seen["calls"] == 1
    assert seen["env"]["POST_MASK_BASE_CONFIG"] == str(tmp_path / "cli.yaml")


def test_cli_entity_mask_root_overrides_the_environment(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _install_recorder(monkeypatch, launcher)
    env_root = _stage2_root(tmp_path, name="env_mask")
    cli_root = _stage2_root(tmp_path, name="cli_mask")
    monkeypatch.setenv("POST_MASK_BASE_CONFIG", str(tmp_path / "cfg.yaml"))
    monkeypatch.setenv("POST_MASK_ENTITY_MASK_ROOT", str(env_root))

    launcher.main(
        [
            "--base-config",
            str(tmp_path / "cfg.yaml"),
            "--entity-mask-root",
            str(cli_root),
            "--job-runner",
            "pkg.module:callable",
        ]
    )

    assert seen["calls"] == 1
    assert seen["env"]["POST_MASK_ENTITY_MASK_ROOT"] == str(cli_root)


def test_explicit_post_mask_root_matches_the_runner_root(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _install_recorder(monkeypatch, launcher)
    root = _stage2_root(tmp_path)
    campaign_root = tmp_path / "workspace" / "data" / "campaign-A"
    monkeypatch.delenv("POST_MASK_ROOT", raising=False)

    launcher.main(
        [
            "--base-config",
            str(tmp_path / "cfg.yaml"),
            "--entity-mask-root",
            str(root),
            "--post-mask-root",
            str(campaign_root),
            "--job-runner",
            "pkg.module:callable",
        ]
    )

    assert seen["calls"] == 1
    assert seen["env"]["POST_MASK_ROOT"] == str(campaign_root)
    assert seen["ledger_root"].parent.parent == campaign_root


def test_env_only_launch_is_unchanged(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _install_recorder(monkeypatch, launcher)
    root = _stage2_root(tmp_path)
    campaign_root = tmp_path / "workspace" / "data" / "campaign-B"
    monkeypatch.setenv("POST_MASK_BASE_CONFIG", str(tmp_path / "env.yaml"))
    monkeypatch.setenv("POST_MASK_ENTITY_MASK_ROOT", str(root))
    monkeypatch.setenv("POST_MASK_TAG", "tag-from-env")
    monkeypatch.setenv("POST_MASK_ROOT", str(campaign_root))

    launcher.main(
        [
            "--base-config",
            str(tmp_path / "env.yaml"),
            "--entity-mask-root",
            str(root),
            "--tag",
            "tag-from-env",
            "--post-mask-root",
            str(campaign_root),
            "--job-runner",
            "pkg.module:callable",
        ]
    )

    assert seen["calls"] == 1
    assert seen["env"]["POST_MASK_BASE_CONFIG"] == str(tmp_path / "env.yaml")
    assert seen["env"]["POST_MASK_TAG"] == "tag-from-env"
    assert seen["ledger_root"].parent.parent == campaign_root


def test_stale_post_mask_root_in_environment_is_not_inherited(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No explicit root anywhere: the runner must not inherit a stale value."""
    seen = _install_recorder(monkeypatch, launcher)
    root = _stage2_root(tmp_path)
    monkeypatch.setenv("POST_MASK_BASE_CONFIG", str(tmp_path / "cfg.yaml"))
    monkeypatch.setenv("POST_MASK_ENTITY_MASK_ROOT", str(root))
    monkeypatch.setenv("POST_MASK_ROOT", str(tmp_path / "stale"))

    launcher.main(
        [
            "--base-config",
            str(tmp_path / "cfg.yaml"),
            "--entity-mask-root",
            str(root),
            "--job-runner",
            "pkg.module:callable",
        ]
    )

    assert seen["calls"] == 1
    assert seen["env"]["POST_MASK_ROOT"] is None
    # The launcher derived its state root from the tag, and so did the runner.
    assert seen["ledger_root"].parent.parent.name == launcher.DEFAULT_TAG


# ---------------------------------------------------------------------------
# Completed-group metadata fast skip and single canonical enumeration
# ---------------------------------------------------------------------------


def _explode_ledger(monkeypatch: pytest.MonkeyPatch) -> None:
    """A completed group must never construct a GroupLedger."""
    import r2v_data_v2.v3.post_mask_epoch_state as state_module

    class _ExplodingLedger:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("a completed group must not construct a GroupLedger")

    monkeypatch.setattr(state_module, "GroupLedger", _ExplodingLedger)


def _summary(campaign_root: Path, rank: int = 0) -> dict[str, Any]:
    path = campaign_root / "resource_epochs" / f"summary-rank{rank}.json"
    return json.loads(path.read_text())


def test_launcher_enumerates_canonical_shards_once(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Startup enumerates once and publishes the count for every group runner."""
    import r2v_data_v2.v3.post_mask_production as production_module

    seen = _install_recorder(monkeypatch, launcher)
    root = _stage2_root(tmp_path, shards=9)
    calls = {"n": 0}
    real_enumerate = production_module.enumerate_shards

    def counting(root_arg: Path) -> Any:
        calls["n"] += 1
        return real_enumerate(root_arg)

    monkeypatch.setattr(production_module, "enumerate_shards", counting)

    assert (
        launcher.main(
            [
                "--base-config",
                str(tmp_path / "cfg.yaml"),
                "--entity-mask-root",
                str(root),
                "--post-mask-root",
                str(tmp_path / "campaign"),
                "--job-runner",
                "pkg.module:callable",
            ]
        )
        == 1
    )
    assert calls["n"] == 1, calls
    assert seen["env"]["POST_MASK_CANONICAL_SHARD_COUNT"] == "9"


def test_completed_group_restart_skips_the_runner_and_the_ledger(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A completed group restart is answered from top-level metadata alone."""
    root = _stage2_root(tmp_path, shards=9)
    campaign_root = tmp_path / "campaign"
    argv = [
        "--base-config",
        str(tmp_path / "cfg.yaml"),
        "--entity-mask-root",
        str(root),
        "--post-mask-root",
        str(campaign_root),
        "--job-runner",
        "pkg.module:callable",
    ]

    seen = _install_recorder(monkeypatch, launcher, completed=True)
    assert launcher.main(argv) == 0
    assert seen["calls"] == 2
    epochs = campaign_root / "resource_epochs"
    for group_id in ("group-000000", "group-000001"):
        marker = epochs / group_id / "COMPLETE.json"
        assert marker.is_file(), group_id
        payload = json.loads(marker.read_text())
        assert payload == {
            "group_id": group_id,
            "group_identity": payload["group_identity"],
        }
        assert len(payload["group_identity"]) == 64
        # The run history is recorded as well as the marker.
        outcomes = [
            json.loads(line)
            for line in (epochs / group_id / "outcomes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert [item["outcome"] for item in outcomes] == ["completed"]

    _explode_ledger(monkeypatch)
    seen["calls"] = 0
    seen["order"] = []
    assert launcher.main(argv) == 0
    assert seen["calls"] == 0, "a completed group never reaches the runner"
    summary = _summary(campaign_root)
    assert summary["groups_complete_skipped"] == 2
    assert summary["groups_unfinished"] == 0
    assert summary["groups_fresh"] == 0
    assert summary["groups_attempted"] == 0


def test_unfinished_groups_are_attempted_before_fresh_groups(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Existing incomplete work is finished before new work is started."""
    root = _stage2_root(tmp_path, shards=9)
    campaign_root = tmp_path / "campaign"
    # group-000001 has a root but no completion marker; group-000000 never ran.
    (campaign_root / "resource_epochs" / "group-000001").mkdir(parents=True)
    seen = _install_recorder(monkeypatch, launcher)

    assert (
        launcher.main(
            [
                "--base-config",
                str(tmp_path / "cfg.yaml"),
                "--entity-mask-root",
                str(root),
                "--post-mask-root",
                str(campaign_root),
                "--job-runner",
                "pkg.module:callable",
            ]
        )
        == 1
    )
    assert seen["order"] == ["group-000001", "group-000000"]
    summary = _summary(campaign_root)
    assert summary["groups_unfinished"] == 1
    assert summary["groups_fresh"] == 1
    assert summary["groups_complete_skipped"] == 0


def test_completed_group_is_recognised_under_a_new_world_size(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Completion is static-ownership independent: a new topology skips it too."""
    root = _stage2_root(tmp_path, shards=9)
    campaign_root = tmp_path / "campaign"
    base = [
        "--base-config",
        str(tmp_path / "cfg.yaml"),
        "--entity-mask-root",
        str(root),
        "--post-mask-root",
        str(campaign_root),
        "--job-runner",
        "pkg.module:callable",
    ]
    seen = _install_recorder(monkeypatch, launcher, completed=True)
    assert launcher.main([*base, "--rank", "0", "--world-size", "1"]) == 0
    assert seen["calls"] == 2

    _explode_ledger(monkeypatch)
    seen["calls"] = 0
    # Under world_size=5 rank=1 owns exactly group-000001, already complete.
    assert launcher.main([*base, "--rank", "1", "--world-size", "5"]) == 0
    assert seen["calls"] == 0
    summary = _summary(campaign_root, rank=1)
    assert summary["groups_complete_skipped"] == 1
    assert summary["groups_attempted"] == 0


def test_mismatched_complete_marker_fails_closed(
    launcher: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A marker that describes another campaign is never treated as absent."""
    root = _stage2_root(tmp_path)
    campaign_root = tmp_path / "campaign"
    seen = _install_recorder(monkeypatch, launcher)
    group_root = campaign_root / "resource_epochs" / "group-000000"
    group_root.mkdir(parents=True)
    (group_root / "COMPLETE.json").write_text(
        json.dumps({"group_id": "group-000000", "group_identity": "0" * 64})
    )

    assert (
        launcher.main(
            [
                "--base-config",
                str(tmp_path / "cfg.yaml"),
                "--entity-mask-root",
                str(root),
                "--post-mask-root",
                str(campaign_root),
                "--job-runner",
                "pkg.module:callable",
            ]
        )
        == 2
    )
    assert seen["calls"] == 0
    assert "completion marker identity mismatch" in capsys.readouterr().out


def test_group_completed_after_the_resume_scan_is_not_rerun(
    launcher: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The resume plan is lock-free, so completion must be rechecked under the lock.

    Another node can finish a group between this node's resume scan and its lock
    acquisition. Trusting the stale plan would re-run a group that is already
    done, so the completion marker is consulted again *after* the lock is held,
    still as metadata only, and the group never reaches the runner.
    """
    from contextlib import contextmanager

    import r2v_data_v2.v3.post_mask_epoch_groups as groups_module

    root = _stage2_root(tmp_path, shards=1)
    campaign_root = tmp_path / "campaign"
    seen = _install_recorder(monkeypatch, launcher)
    _explode_ledger(monkeypatch)

    real_ownership = groups_module.group_ownership
    raced: list[str] = []

    @contextmanager
    def racing_ownership(state_root: Path, group: Any):
        # The simulated peer publishes its completion before we take the lock.
        marker = groups_module.group_completed_marker(state_root, group)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {"group_id": group.group_id, "group_identity": group.identity()}
            )
        )
        raced.append(group.group_id)
        with real_ownership(state_root, group) as held:
            yield held

    monkeypatch.setattr(groups_module, "group_ownership", racing_ownership)

    assert (
        launcher.main(
            [
                "--base-config",
                str(tmp_path / "cfg.yaml"),
                "--entity-mask-root",
                str(root),
                "--post-mask-root",
                str(campaign_root),
                "--job-runner",
                "pkg.module:callable",
            ]
        )
        == 0
    )
    assert raced == ["group-000000"]
    assert seen["calls"] == 0, "a group completed elsewhere never reaches the runner"
    summary = _summary(campaign_root)
    assert summary["groups_attempted"] == 0, "it is not counted as attempted"
    assert summary["groups_complete_skipped"] == 1
    assert summary["groups_complete_skipped_after_lock"] == 1
    assert summary["group_completed"] == 0


def test_started_event_reports_the_execution_only_cpu_budget(
    launcher: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The worker budget is reported but never enters campaign identity."""
    root = _stage2_root(tmp_path)
    monkeypatch.setenv("POST_MASK_CPU_WORKERS", "32")
    _install_recorder(monkeypatch, launcher, completed=True)

    assert (
        launcher.main(
            [
                "--base-config",
                str(tmp_path / "cfg.yaml"),
                "--entity-mask-root",
                str(root),
                "--post-mask-root",
                str(tmp_path / "campaign"),
                "--job-runner",
                "pkg.module:callable",
            ]
        )
        == 0
    )

    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("{")
    ]
    started = next(
        item
        for item in events
        if item.get("event") == "post_mask_resource_epoch_started"
    )
    assert started["cpu_workers"] == 32
    # Campaign identity is unchanged by an execution-only setting.
    campaign = launcher.build_campaign(
        config_module.load_config(Path(str(tmp_path / "cfg.yaml"))),
        entity_mask_root=root,
        canonical_shard_count=1,
    )
    monkeypatch.setenv("POST_MASK_CPU_WORKERS", "8")
    again = launcher.build_campaign(
        config_module.load_config(Path(str(tmp_path / "cfg.yaml"))),
        entity_mask_root=root,
        canonical_shard_count=1,
    )
    assert campaign == again
