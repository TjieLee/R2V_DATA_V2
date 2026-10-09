"""Focused filesystem-race regressions for durable resource-epoch plans."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from r2v_data_v2.v3.post_mask_epoch_jobs import RESOURCE_QWEN, JobResult, ModelJob
from r2v_data_v2.v3.post_mask_epoch_state import (
    GroupLedger,
    LedgerError,
    PhaseLedger,
    PlanMismatchError,
    Receipt,
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


def _commit(phase: PhaseLedger, job: ModelJob, outcome: str = "completed") -> Receipt:
    result = JobResult(outcome, payload={"status": "review", "verdict": "accept"})
    digest = phase.publish_result(job, result)
    return phase.commit(job, outcome=outcome, artifact_digests={}, result_digest=digest)


def test_unique_replay_refreshes_only_requested_job(tmp_path: Path):
    ledger = GroupLedger(tmp_path / "group")
    requested, unrelated = _job("requested"), _job("unrelated")
    ledger.refresh()
    refreshes = ledger.refresh_calls
    phase = ledger.phase("r000-qwen")
    _commit(phase, requested)
    _commit(phase, unrelated)

    result = ledger.replay_unique_committed(requested)

    assert result is not None
    assert result.payload == {"status": "review", "verdict": "accept"}
    assert ledger.classify(requested).state == "completed"
    assert ledger.phase_for(requested) == "r000-qwen"
    assert ledger.classify(unrelated).state == "pending"
    assert ledger.refresh_calls == refreshes


@pytest.mark.parametrize("second_phase", ["r000-qwen", "r001-qwen"])
def test_unique_replay_accepts_identical_committed_duplicates(
    tmp_path: Path, second_phase: str
):
    ledger = GroupLedger(tmp_path / "group")
    job = _job("clip-1")
    _commit(ledger.phase("r000-qwen"), job)
    _commit(ledger.phase(second_phase), job)

    result = ledger.replay_unique_committed(job)

    assert result is not None
    assert result.payload["verdict"] == "accept"


@pytest.mark.parametrize("second_phase", ["r000-qwen", "r001-qwen"])
@pytest.mark.parametrize("field,value", [
    ("job_identity", "different-identity"),
    ("outcome", "terminal_reject"),
    ("result_digest", "different-result"),
    ("artifact_digests", {"mask.npy": "different-mask"}),
    ("external_artifacts", [{"path": "/different.png", "sha256": "different"}]),
    ("model_identity", "different-model"),
])
def test_unique_replay_rejects_contradictory_committed_history(
    tmp_path: Path, second_phase: str, field: str, value: Any
):
    ledger = GroupLedger(tmp_path / "group")
    job = _job("clip-1")
    first = _commit(ledger.phase("r000-qwen"), job)
    ledger.phase(second_phase).append_receipt(
        Receipt.from_record({**first.record(), field: value})
    )

    with pytest.raises(LedgerError, match="contradictory committed receipts"):
        ledger.replay_unique_committed(job)


def test_unique_replay_allows_retryable_history_before_commit(tmp_path: Path):
    ledger = GroupLedger(tmp_path / "group")
    job = _job("clip-1")
    phase = ledger.phase("r000-qwen")
    _commit(phase, job, "retryable_failed")
    _commit(phase, job)

    result = ledger.replay_unique_committed(job)

    assert result is not None
    assert result.outcome == "completed"


@pytest.mark.parametrize("orphan_result", [False, True])
def test_unique_replay_never_commits_pending_or_orphan_result(
    tmp_path: Path, orphan_result: bool
):
    ledger = GroupLedger(tmp_path / "group")
    job = _job("clip-1")
    ledger.refresh()
    if orphan_result:
        ledger.phase("r000-qwen").publish_result(job, JobResult("completed"))

    assert ledger.replay_unique_committed(job) is None
    assert ledger.classify(job).state == "pending"


@pytest.mark.parametrize("damage", ["missing", "invalid", "changed_payload", "mask"])
def test_unique_replay_audits_existing_committed_evidence(
    tmp_path: Path, damage: str
):
    ledger = GroupLedger(tmp_path / "group")
    job = _job("clip-1")
    phase = ledger.phase("r000-qwen")
    _commit(phase, job)
    result_path = phase.artifact_path(job.job_id(), "result.json")
    if damage == "missing":
        result_path.unlink()
    elif damage == "invalid":
        result_path.write_text("not json")
    elif damage == "changed_payload":
        phase.publish_result(job, JobResult("completed", payload={"verdict": "reject"}))
    else:
        phase.publish_artifact(job.job_id(), "unexpected-mask.npy", b"mask")
        receipt = phase.receipts()[job.job_id()]
        receipt["artifact_digests"] = {"expected-mask.npy": "not-on-disk"}
        phase.receipts_path.write_text(json.dumps(receipt) + "\n")

    with pytest.raises(LedgerError):
        ledger.replay_unique_committed(job)


def test_unique_replay_clears_a_stale_requested_commit_when_now_pending(tmp_path: Path):
    ledger = GroupLedger(tmp_path / "group")
    job = _job("clip-1")
    phase = ledger.phase("r000-qwen")
    _commit(phase, job)
    assert ledger.classify(job).skippable
    phase.receipts_path.write_text("")

    assert ledger.replay_unique_committed(job) is None
    assert ledger.classify(job).state == "pending"


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
