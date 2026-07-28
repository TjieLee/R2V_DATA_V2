from __future__ import annotations

from pathlib import Path

import pytest

import r2v_data_v2.config as config_module


@pytest.fixture(autouse=True)
def _local_path_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config_module, "ALLOWED_OUTPUT_ROOT", tmp_path.resolve())
    monkeypatch.setattr(config_module, "ALLOWED_DATASET_ROOT", tmp_path.resolve())
