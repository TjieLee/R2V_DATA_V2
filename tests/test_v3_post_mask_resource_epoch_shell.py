"""Shell-level tests for the opt-in resource-epoch launcher wrapper.

These run ``scripts/run_v3_post_mask_resource_epoch.sh`` for real, against a
temporary fake repo whose ``.venv/bin/python`` only records its argv. That is
the only way to prove what the wrapper actually passes to Python: the Python
``main()`` tests cannot see a flag the shell never emitted, and a shell that
emitted both ``--dry-run`` and ``--job-runner`` would already have been
rejected by argparse before ``main()`` could be blamed.

No model, GPU, endpoint or real interpreter is started.
"""

from __future__ import annotations

import os
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_v3_post_mask_resource_epoch.sh"


@pytest.fixture
def fake_repo(tmp_path: Path) -> tuple[Path, Path]:
    """A checkout-shaped directory whose python records argv instead of running."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    recorded = repo / "argv.txt"
    python = venv_bin / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" > "{recorded}"\n',
        encoding="utf-8",
    )
    python.chmod(python.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return repo, recorded


def _run(
    repo: Path,
    *args: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the wrapper with every POST_MASK_* variable cleared by default."""
    child_env = {
        key: value for key, value in os.environ.items() if not key.startswith("POST_MASK_")
    }
    child_env["POST_MASK_REPO"] = str(repo)
    child_env.update(env or {})
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
    )


def _argv(recorded: Path) -> list[str]:
    return [line for line in recorded.read_text(encoding="utf-8").splitlines() if line]


def _option_values(argv: Sequence[str], flag: str) -> list[str]:
    return [argv[index + 1] for index, value in enumerate(argv) if value == flag]


def test_no_runner_anywhere_falls_back_to_dry_run(
    fake_repo: tuple[Path, Path]
) -> None:
    repo, recorded = fake_repo

    result = _run(repo, "--base-config", "/tmp/cfg.yaml")

    assert result.returncode == 0
    argv = _argv(recorded)
    assert "--dry-run" in argv
    assert "--job-runner" not in argv


def test_environment_runner_is_passed_through(
    fake_repo: tuple[Path, Path]
) -> None:
    repo, recorded = fake_repo

    result = _run(
        repo,
        "--base-config",
        "/tmp/cfg.yaml",
        env={"POST_MASK_JOB_RUNNER": "pkg.module:runner_a"},
    )

    assert result.returncode == 0
    argv = _argv(recorded)
    assert _option_values(argv, "--job-runner") == ["pkg.module:runner_a"]
    assert "--dry-run" not in argv


def test_cli_dry_run_beats_the_environment_runner(
    fake_repo: tuple[Path, Path]
) -> None:
    repo, recorded = fake_repo

    result = _run(
        repo,
        "--base-config",
        "/tmp/cfg.yaml",
        "--dry-run",
        env={"POST_MASK_JOB_RUNNER": "pkg.module:runner_a"},
    )

    assert result.returncode == 0
    argv = _argv(recorded)
    # The mutually exclusive pair must never both reach the parser.
    assert "--dry-run" in argv
    assert "--job-runner" not in argv


def test_cli_runner_beats_the_environment_runner(
    fake_repo: tuple[Path, Path]
) -> None:
    repo, recorded = fake_repo

    result = _run(
        repo,
        "--base-config",
        "/tmp/cfg.yaml",
        "--job-runner",
        "pkg.module:runner_b",
        env={"POST_MASK_JOB_RUNNER": "pkg.module:runner_a"},
    )

    assert result.returncode == 0
    argv = _argv(recorded)
    # Exactly one --job-runner, and it is the CLI one.
    assert _option_values(argv, "--job-runner") == ["pkg.module:runner_b"]
    assert "--dry-run" not in argv


def test_cli_only_base_config_is_accepted(
    fake_repo: tuple[Path, Path]
) -> None:
    repo, recorded = fake_repo

    result = _run(repo, "--base-config", "/tmp/cli-only.yaml")

    assert result.returncode == 0
    assert "is required" not in result.stderr
    argv = _argv(recorded)
    assert _option_values(argv, "--base-config") == ["/tmp/cli-only.yaml"]


def test_no_base_config_anywhere_still_fails(
    fake_repo: tuple[Path, Path]
) -> None:
    repo, _ = fake_repo

    result = _run(repo)

    assert result.returncode == 2
    assert "is required" in result.stderr
