"""Durable plans, receipts and crash recovery for resource-epoch execution.

Layout below the existing Post-Mask campaign state root:

    <post-mask-state>/resource_epochs/group-000000/
        group.json
        phases/<phase-id>/
            plan.json
            receipts.jsonl
            artifacts/<...>

The commit protocol is deliberately strict so that an interruption at any point
can only ever lose *one* job's worth of work, never corrupt a finished one:

    model call
    -> write output to a unique same-directory .tmp
    -> fsync
    -> atomic rename
    -> validate output
    -> append durable receipt
    -> flush + fsync

Consequences used by :meth:`PhaseLedger.classify`:

* valid receipt whose artifacts still match -> completed, skip the model call;
* ``terminal_reject`` receipt -> durable quality outcome, never paid for again;
* artifact present but no valid receipt -> the job did not commit, rerun it;
* receipt/artifact digest mismatch -> fail closed with a diagnostic, never
  silently accept;
* absent receipt and absent artifact -> ordinary pending job.

History is append-only. Diagnostics are never deleted. The only mutation ever
performed on ``receipts.jsonl`` is truncating a single torn **final** line left
by an interrupted append, matching the project's existing JSONL pattern.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2v_data_v2.v3.post_mask_epoch_jobs import (
    COMMITTED_OUTCOMES,
    OUTCOME_COMPLETED,
    canonical_json,
)

_PLAN_NAME = "plan.json"
_RECEIPTS_NAME = "receipts.jsonl"
_ARTIFACTS_NAME = "artifacts"

# Resume classification results.
STATE_PENDING = "pending"
STATE_COMPLETED = "completed"
STATE_TERMINAL_REJECT = "terminal_reject"
STATE_RERUN_NO_RECEIPT = "rerun_no_receipt"
STATE_MISMATCH = "mismatch"

_SKIPPABLE = (STATE_COMPLETED, STATE_TERMINAL_REJECT)


class LedgerError(RuntimeError):
    """Durable ledger could not be used safely; fail closed."""


class PlanMismatchError(LedgerError):
    """An existing immutable plan disagrees with the recomputed plan."""


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync; unsupported platforms are not production."""
    try:
        handle = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(handle)
    except OSError:
        pass
    finally:
        os.close(handle)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish ``payload`` atomically inside ``path``'s own directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, (canonical_json(payload) + "\n").encode("utf-8"))


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Return parsed records and whether a torn final line was repaired.

    Only the final line may be truncated. A malformed *earlier* line is a real
    integrity problem and raises instead of being silently dropped.
    """
    if not path.is_file():
        return [], False
    raw = path.read_bytes()
    if not raw:
        return [], False
    repaired = False
    if not raw.endswith(b"\n"):
        # The writer appends a complete line plus "\n" in one write, so a
        # missing terminator means the append was interrupted mid-line.
        cut = raw.rfind(b"\n")
        raw = raw[: cut + 1] if cut >= 0 else b""
        repaired = True
        atomic_write_bytes(path, raw)
    records: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise LedgerError(f"corrupt receipt line {number} in {path}") from exc
        if not isinstance(record, dict):
            raise LedgerError(f"receipt line {number} in {path} is not an object")
        records.append(record)
    return records, repaired


@dataclass(frozen=True)
class JobState:
    """Resume decision for one planned job."""

    state: str
    receipt: dict[str, Any] | None = None
    detail: str = ""

    @property
    def skippable(self) -> bool:
        return self.state in _SKIPPABLE


@dataclass(frozen=True)
class Receipt:
    job_id: str
    job_identity: str
    resource: str
    outcome: str
    artifact_digests: Mapping[str, str]
    model_identity: str

    def record(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_identity": self.job_identity,
            "resource": self.resource,
            "outcome": self.outcome,
            "artifact_digests": dict(sorted(self.artifact_digests.items())),
            "model_identity": self.model_identity,
        }


class PhaseLedger:
    """Durable, resumable state for one resource phase of one group."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.artifacts_root = self.root / _ARTIFACTS_NAME
        self.plan_path = self.root / _PLAN_NAME
        self.receipts_path = self.root / _RECEIPTS_NAME
        self.torn_receipts_repaired = 0

    # -- plan -------------------------------------------------------------
    def plan_hash(self, jobs: Sequence[Any]) -> str:
        payload = [job.plan_record() for job in jobs]
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def write_plan(self, jobs: Sequence[Any]) -> str:
        """Create or validate the immutable phase plan; never overwrite."""
        digest = self.plan_hash(jobs)
        payload = {
            "plan_hash": digest,
            "job_count": len(jobs),
            "jobs": [job.plan_record() for job in jobs],
        }
        if self.plan_path.is_file():
            existing = json.loads(self.plan_path.read_text())
            if existing != payload:
                raise PlanMismatchError(
                    f"immutable phase plan changed at {self.plan_path}"
                )
            return digest
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.plan_path, payload)
        return digest

    def read_plan(self) -> list[dict[str, Any]]:
        if not self.plan_path.is_file():
            raise LedgerError(f"missing phase plan at {self.plan_path}")
        payload = json.loads(self.plan_path.read_text())
        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            raise LedgerError(f"invalid phase plan at {self.plan_path}")
        return jobs

    # -- receipts ---------------------------------------------------------
    def receipts(self) -> dict[str, dict[str, Any]]:
        records, repaired = _load_jsonl(self.receipts_path)
        if repaired:
            self.torn_receipts_repaired += 1
        latest: dict[str, dict[str, Any]] = {}
        for record in records:
            key = record.get("job_id")
            if isinstance(key, str):
                latest[key] = record
        return latest

    def append_receipt(self, receipt: Receipt) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        line = canonical_json(receipt.record()) + "\n"
        with self.receipts_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())

    # -- artifacts --------------------------------------------------------
    def artifact_path(self, job_id: str, name: str) -> Path:
        return self.artifacts_root / job_id / name

    def publish_artifact(
        self,
        job_id: str,
        name: str,
        payload: bytes,
        *,
        validate: Callable[[bytes], None] | None = None,
    ) -> str:
        """Atomically publish one artifact and return its digest."""
        target = self.artifact_path(job_id, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if validate is not None:
                validate(temporary.read_bytes())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        _fsync_directory(target.parent)
        return hashlib.sha256(payload).hexdigest()

    def artifact_digests(self, job_id: str) -> dict[str, str]:
        root = self.artifacts_root / job_id
        if not root.is_dir():
            return {}
        result = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                result[str(path.relative_to(root))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        return result

    # -- resume -----------------------------------------------------------
    def classify(self, job: Any) -> JobState:
        receipt = self.receipts().get(job.job_id())
        if receipt is None:
            return JobState(
                STATE_RERUN_NO_RECEIPT
                if self.artifact_digests(job.job_id())
                else STATE_PENDING,
                detail="artifact without receipt"
                if self.artifact_digests(job.job_id())
                else "",
            )
        if receipt.get("job_identity") != job.identity():
            return JobState(
                STATE_MISMATCH,
                receipt,
                "receipt job identity does not match the planned job",
            )
        outcome = receipt.get("outcome")
        if outcome not in COMMITTED_OUTCOMES:
            return JobState(STATE_PENDING, receipt, "receipt is not a committed outcome")
        if outcome == OUTCOME_COMPLETED:
            expected = receipt.get("artifact_digests") or {}
            actual = self.artifact_digests(job.job_id())
            if expected != actual:
                return JobState(
                    STATE_MISMATCH,
                    receipt,
                    "receipt artifact digests do not match artifacts on disk",
                )
            return JobState(STATE_COMPLETED, receipt)
        return JobState(STATE_TERMINAL_REJECT, receipt)

    def commit(
        self,
        job: Any,
        *,
        outcome: str,
        artifact_digests: Mapping[str, str],
    ) -> Receipt:
        receipt = Receipt(
            job_id=job.job_id(),
            job_identity=job.identity(),
            resource=job.resource,
            outcome=outcome,
            artifact_digests=dict(artifact_digests),
            model_identity=job.model_identity,
        )
        self.append_receipt(receipt)
        return receipt


class GroupLedger:
    """Aggregate every phase ledger of one resource epoch group.

    A conditional second attempt is planned in a later phase than the job that
    unlocked it, so resume has to locate a receipt without knowing in advance
    which phase produced it. Phases are scanned in lexical order and the newest
    matching receipt wins; an artifact/receipt mismatch is still evaluated
    against the phase that owns the artifact.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def phases_root(self) -> Path:
        return self.root / "phases"

    def phase(self, phase_id: str) -> PhaseLedger:
        return PhaseLedger(self.phases_root / phase_id)

    def phase_ids(self) -> tuple[str, ...]:
        if not self.phases_root.is_dir():
            return ()
        return tuple(
            sorted(path.name for path in self.phases_root.iterdir() if path.is_dir())
        )

    def _index(self) -> dict[str, tuple[str, dict[str, Any]]]:
        index: dict[str, tuple[str, dict[str, Any]]] = {}
        for phase_id in self.phase_ids():
            for job_id, receipt in self.phase(phase_id).receipts().items():
                index[job_id] = (phase_id, receipt)
        return index

    def classify(self, job: Any) -> JobState:
        entry = self._index().get(job.job_id())
        if entry is None:
            for phase_id in self.phase_ids():
                if self.phase(phase_id).artifact_digests(job.job_id()):
                    return JobState(
                        STATE_RERUN_NO_RECEIPT, detail="artifact without receipt"
                    )
            return JobState(STATE_PENDING)
        phase_id, receipt = entry
        state = self.phase(phase_id).classify(job)
        if state.state == STATE_PENDING and not state.detail:
            return JobState(STATE_PENDING, receipt, f"located in {phase_id}")
        return state

    @property
    def torn_receipts_repaired(self) -> int:
        return sum(self.phase(pid).torn_receipts_repaired for pid in self.phase_ids())


@contextmanager
def file_lock(path: Path, *, blocking: bool = True) -> Iterator[bool]:
    """Exclusive advisory lock on a dedicated lock file.

    Production runs on Linux and uses ``fcntl.flock``. A narrow ``msvcrt``
    fallback exists only so this module stays unit-testable on Windows; it is
    not a substitute for POSIX advisory locking and must not be relied on for
    cross-node correctness.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        try:
            import fcntl

            operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), operation)
        except ImportError:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        except OSError:
            handle.close()
            yield False
            return
        try:
            yield True
        finally:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
    finally:
        handle.close()
