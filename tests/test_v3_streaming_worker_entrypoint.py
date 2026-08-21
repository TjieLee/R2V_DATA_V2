from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_streaming_stage_worker_bootstraps_repository_import_path(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    worker = repository_root / "tools" / "run_v3_streaming_stage_worker.py"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"

    completed = subprocess.run(
        [sys.executable, str(worker), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--stage" in completed.stdout
    assert "--attribute-probe-only" in completed.stdout
