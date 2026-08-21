from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_v3_subject_attribute_backfill as backfill


def test_backfill_reuses_visual_run_and_existing_attribute_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = (tmp_path / "run").resolve()
    config = SimpleNamespace(
        resolved_run_root=run_root,
        runtime=SimpleNamespace(
            gpu_workers=SimpleNamespace(subject_attributes_segment="7")
        ),
        subject_attribute_gme=SimpleNamespace(enabled=False),
    )
    observed: dict[str, object] = {}

    def enrich(received: object, **kwargs: object) -> dict[str, object]:
        observed["config"] = received
        observed.update(kwargs)
        observed["visible_gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES")
        return {"accepted_attributes": 3}

    monkeypatch.setattr(backfill, "run_subject_attribute_enrichment", enrich)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "original")

    result = backfill.run_backfill(config)  # type: ignore[arg-type]

    assert result == {"accepted_attributes": 3}
    assert observed["config"] is config
    assert observed["run_root"] == run_root
    assert observed["output_root"] == run_root / "subject_attributes"
    assert observed["max_owners"] is None
    assert observed["overwrite"] is False
    assert observed["allow_run_local_sidecar"] is True
    assert observed["visible_gpu"] == "7"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "original"
    assert config.subject_attribute_gme.enabled is False


def test_backfill_requires_existing_attribute_gpu_assignment(tmp_path: Path) -> None:
    config = SimpleNamespace(
        resolved_run_root=(tmp_path / "run").resolve(),
        runtime=SimpleNamespace(
            gpu_workers=SimpleNamespace(subject_attributes_segment=None)
        ),
    )

    with pytest.raises(ValueError, match="subject_attributes_segment is required"):
        backfill.run_backfill(config)  # type: ignore[arg-type]
