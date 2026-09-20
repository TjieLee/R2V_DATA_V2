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


def _stage2_root(tmp_path: Path, name: str = "entity_mask") -> Path:
    root = tmp_path / name
    parts = root / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    (parts / f"{SHARD}.jsonl").write_text("")
    return root


def _install_recorder(
    monkeypatch: pytest.MonkeyPatch, launcher: Any
) -> dict[str, Any]:
    """Replace the job runner loader with one that records what it observes."""
    seen: dict[str, Any] = {"loads": 0, "calls": 0}

    def fake_load(spec: str) -> Any:
        seen["loads"] += 1
        seen["spec"] = spec

        def runner(group: Any, ledger: Any, emit: Any) -> dict[str, Any]:
            seen["calls"] += 1
            seen["group_id"] = getattr(group, "group_id", "")
            seen["ledger_root"] = Path(ledger.root)
            seen["env"] = {
                name: os.environ.get(name)
                for name in (
                    "POST_MASK_BASE_CONFIG",
                    "POST_MASK_TAG",
                    "POST_MASK_ENTITY_MASK_ROOT",
                    "POST_MASK_ROOT",
                    "POST_MASK_JOB_RUNNER",
                )
            }
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
