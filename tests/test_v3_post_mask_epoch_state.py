"""Focused filesystem-race regressions for durable resource-epoch plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN, ModelJob
from r2v_data_v2.v3.post_mask_epoch_state import (
    LedgerError,
    PhaseLedger,
    PlanMismatchError,
)


def _job(clip_uid: str) -> ModelJob:
    return ModelJob.create(
        job_type="subject_attributes",
        resource=RESOURCE_QWEN,
        canonical_shard="shard-000000000-000000999",
        clip_uid=clip_uid,
        semantic_inputs={"prompt": "describe subject"},
        model_identity="fake-qwen",
    )


def test_existing_plan_retries_missing_read_without_shrinking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ledger = PhaseLedger(tmp_path / "phase")
    first, second = _job("clip-1"), _job("clip-2")
    ledger.write_plan([first, second])
    original = ledger.plan_path.read_bytes()
    read_text = Path.read_text
    reads = 0

    def transient_missing(path: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal reads
        if path == ledger.plan_path:
            reads += 1
            if reads == 1:
                raise FileNotFoundError(str(path))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient_missing)
    ledger.write_plan([second])

    assert reads == 2
    assert ledger.plan_path.read_bytes() == original
    assert {record["job_id"] for record in ledger.read_plan()} == {
        first.job_id(), second.job_id()
    }


def test_existing_plan_persistently_missing_read_fails_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    ledger = PhaseLedger(tmp_path / "phase")
    first, second = _job("clip-1"), _job("clip-2")
    ledger.write_plan([first, second])
    original = ledger.plan_path.read_bytes()
    read_text = Path.read_text
    reads = 0

    def persistent_missing(path: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal reads
        if path == ledger.plan_path:
            reads += 1
            raise FileNotFoundError(str(path))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", persistent_missing)
    with pytest.raises(LedgerError, match="plan") as error:
        ledger.write_plan([second])

    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert reads == 3
    assert ledger.plan_path.read_bytes() == original


def test_absent_plan_creation_and_monotonic_growth_are_unchanged(tmp_path: Path):
    ledger = PhaseLedger(tmp_path / "phase")
    first, second = _job("clip-1"), _job("clip-2")

    ledger.write_plan([first])
    assert ledger.read_plan() == [first.plan_record()]
    ledger.write_plan([second])
    grown = ledger.plan_path.read_bytes()
    assert {record["job_id"] for record in ledger.read_plan()} == {
        first.job_id(), second.job_id()
    }
    ledger.write_plan([first])
    assert ledger.plan_path.read_bytes() == grown


@pytest.mark.parametrize("invalid", ["json", "schema", "identity"])
def test_read_retry_does_not_hide_invalid_existing_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
):
    ledger = PhaseLedger(tmp_path / "phase")
    job = _job("clip-1")
    ledger.write_plan([job])
    current: Any = job
    error_type: type[Exception]
    if invalid == "json":
        ledger.plan_path.write_text("not json")
        error_type = json.JSONDecodeError
    elif invalid == "schema":
        ledger.plan_path.write_text(json.dumps({"jobs": "not a list"}))
        error_type = LedgerError
    else:
        class ChangedJob:
            def job_id(self) -> str:
                return job.job_id()

            def plan_record(self) -> dict[str, Any]:
                return {**job.plan_record(), "job_type": "changed"}

        current = ChangedJob()
        error_type = PlanMismatchError
    original = ledger.plan_path.read_bytes()
    read_text = Path.read_text
    reads = 0

    def transient_missing(path: Path, *args: Any, **kwargs: Any) -> str:
        nonlocal reads
        if path == ledger.plan_path:
            reads += 1
            if reads == 1:
                raise FileNotFoundError(str(path))
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient_missing)
    with pytest.raises(error_type):
        ledger.write_plan([current])

    assert reads == 2
    assert ledger.plan_path.read_bytes() == original
