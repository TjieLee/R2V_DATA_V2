"""Durable plans, receipts and crash recovery for resource-epoch execution.

Layout below the existing Post-Mask campaign state root:

    <post-mask-state>/resource_epochs/group-000000/
        group.json
        phases/<phase-id>/
            plan.json
            receipts.jsonl
            artifacts/<job-id>/<...>
            artifacts/<job-id>/result.json

The commit protocol is deliberately strict so that an interruption at any point
can only ever lose *one* job's worth of work, never corrupt a finished one:

    model call
    -> write output to a unique same-directory .tmp
    -> fsync
    -> atomic rename
    -> validate output
    -> append durable receipt
    -> flush + fsync

Consequences used by :meth:`GroupLedger.classify`:

* valid receipt whose artifacts still match -> completed, skip the model call
  but still replay the CPU finalizer;
* ``terminal_reject`` receipt -> durable quality outcome, never paid for again,
  finalizer still replayed;
* artifact present but no valid receipt -> the job did not commit, rerun it;
* receipt/artifact digest mismatch -> fail closed with a diagnostic, never
  silently accept;
* absent receipt and absent artifact -> ordinary pending job.

Skipping a model call is **not** the same as skipping the finalizer. A
downstream job only exists because some finalizer created it, so a crash
between the receipt and the finalizer would otherwise strand every dependent
job forever. Committed results are therefore persisted as ``result.json`` and
replayed on resume.

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
    ArtifactReference,
    JobResult,
    canonical_json,
)

_PLAN_NAME = "plan.json"
_RECEIPTS_NAME = "receipts.jsonl"
_ARTIFACTS_NAME = "artifacts"
_RESULT_NAME = "result.json"

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


def verify_external_artifact(
    reference: ArtifactReference, *, chunk_size: int = 1 << 20
) -> bool:
    """Validate a referenced production artifact without copying it.

    ``ArtifactReference.path`` is an internal ledger-facing absolute filesystem
    path, never a public schema path; public relative paths belong in
    ``JobResult.payload``. Reads are chunked so a large generated PNG is never
    pulled into memory in one go.
    """
    digest = hashlib.sha256()
    try:
        path = Path(reference.path)
        if not path.is_absolute():
            return False
        if not path.is_file():
            return False
        if reference.size is not None and path.stat().st_size != reference.size:
            return False
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError:
        # Any filesystem race (disappeared file, permission change, stale NFS
        # handle) fails closed instead of being reported as a valid artifact.
        return False
    return digest.hexdigest() == reference.sha256


@dataclass(frozen=True)
class JobState:
    """Resume decision for one planned job."""

    state: str
    receipt: dict[str, Any] | None = None
    detail: str = ""
    phase_id: str | None = None

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
    result_digest: str = ""
    external_artifacts: tuple[ArtifactReference, ...] = ()

    def record(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_identity": self.job_identity,
            "resource": self.resource,
            "outcome": self.outcome,
            "artifact_digests": dict(sorted(self.artifact_digests.items())),
            "model_identity": self.model_identity,
            "result_digest": self.result_digest,
            "external_artifacts": [ref.record() for ref in self.external_artifacts],
        }

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> Receipt:
        return cls(
            job_id=str(value.get("job_id", "")),
            job_identity=str(value.get("job_identity", "")),
            resource=str(value.get("resource", "")),
            outcome=str(value.get("outcome", "")),
            artifact_digests=dict(value.get("artifact_digests") or {}),
            model_identity=str(value.get("model_identity", "")),
            result_digest=str(value.get("result_digest", "")),
            external_artifacts=tuple(
                ArtifactReference.from_record(item)
                for item in value.get("external_artifacts", [])
            ),
        )


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

    @staticmethod
    def _validate_existing_plan(payload: Any) -> dict[str, dict[str, Any]]:
        """Validate a plan already on disk before it is extended.

        A malformed plan must fail closed. It is never silently repaired or
        overwritten, because that would hide durable-state corruption and
        could resurrect a job under different semantics.
        """
        if not isinstance(payload, dict):
            raise LedgerError("phase plan is not an object")
        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            raise LedgerError("phase plan jobs is not a list")
        count = payload.get("job_count")
        if not isinstance(count, int) or isinstance(count, bool) or count != len(jobs):
            raise LedgerError("phase plan job_count does not match its jobs")
        plan_hash = payload.get("plan_hash")
        if not isinstance(plan_hash, str) or not plan_hash:
            raise LedgerError("phase plan has no plan_hash")
        if (
            hashlib.sha256(canonical_json(jobs).encode("utf-8")).hexdigest()
            != plan_hash
        ):
            raise LedgerError("phase plan hash does not match its jobs")
        indexed: dict[str, dict[str, Any]] = {}
        for record in jobs:
            if not isinstance(record, dict):
                raise LedgerError("phase plan job record is not an object")
            job_id = record.get("job_id")
            if not isinstance(job_id, str) or not job_id:
                raise LedgerError("phase plan job record has no job_id")
            if job_id in indexed:
                raise LedgerError(f"phase plan has duplicate job_id {job_id}")
            if not isinstance(record.get("job_identity"), str):
                raise LedgerError(f"phase plan job {job_id} has no job_identity")
            indexed[job_id] = record
        return indexed

    def write_plan(self, jobs: Sequence[Any]) -> str:
        """Create or grow the phase plan. Never shrink it, never rewrite a job.

        A phase's ready set can grow across a restart: a retryable sibling that
        succeeds on the second run unlocks jobs that did not exist the first
        time. Treating the batch as an immutable whole therefore produced a
        false ``PlanMismatchError``. The invariant that actually matters is
        per-job: a job's ``plan_record`` may never change once written.

        So the plan's job set is monotonic and each record is immutable.
        """
        current = {job.job_id(): job.plan_record() for job in jobs}
        if not self.plan_path.is_file():
            ordered = [current[key] for key in sorted(current)]
            payload = {
                "plan_hash": hashlib.sha256(
                    canonical_json(ordered).encode("utf-8")
                ).hexdigest(),
                "job_count": len(ordered),
                "jobs": ordered,
            }
            self.root.mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.plan_path, payload)
            return payload["plan_hash"]

        existing = json.loads(self.plan_path.read_text())
        merged = self._validate_existing_plan(existing)
        for job_id, record in current.items():
            known = merged.get(job_id)
            if known is not None and known != record:
                raise PlanMismatchError(
                    f"immutable job plan changed for {job_id} at {self.plan_path}"
                )
            merged[job_id] = record
        ordered = [merged[key] for key in sorted(merged)]
        payload = {
            "plan_hash": hashlib.sha256(canonical_json(ordered).encode("utf-8")).hexdigest(),
            "job_count": len(ordered),
            "jobs": ordered,
        }
        if payload != existing:
            atomic_write_json(self.plan_path, payload)
        return payload["plan_hash"]

    def read_plan(self) -> list[dict[str, Any]]:
        if not self.plan_path.is_file():
            raise LedgerError(f"missing phase plan at {self.plan_path}")
        payload = json.loads(self.plan_path.read_text())
        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            raise LedgerError(f"invalid phase plan at {self.plan_path}")
        return jobs

    # -- receipts ---------------------------------------------------------
    def load_receipts(self) -> tuple[list[dict[str, Any]], bool]:
        records, repaired = _load_jsonl(self.receipts_path)
        if repaired:
            self.torn_receipts_repaired += 1
        return records, repaired

    def receipts(self) -> dict[str, dict[str, Any]]:
        records, _ = self.load_receipts()
        latest: dict[str, dict[str, Any]] = {}
        for record in records:
            key = record.get("job_id")
            if isinstance(key, str):
                latest[key] = record
        return latest

    def append_receipt(self, receipt: Receipt) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        was_new = not self.receipts_path.exists()
        line = canonical_json(receipt.record()) + "\n"
        with self.receipts_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        if was_new:
            # A newly created file needs its directory entry durable too,
            # otherwise a crash can lose the file itself, not just its tail.
            _fsync_directory(self.receipts_path.parent)

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

    def artifact_digests(
        self, job_id: str, *, include_result: bool = True
    ) -> dict[str, str]:
        """Digest the artifacts of one job directory.

        ``include_result`` keeps the durable ``result.json`` in the listing. It
        defaults to True because a job with artifacts but no receipt still has to
        be recognised as a rerun rather than as pending. The committed path
        passes False, so the result is never hashed as if it were a binary
        artifact and never hashed twice.
        """
        root = self.artifacts_root / job_id
        if not root.is_dir():
            return {}
        result = {}
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            name = str(path.relative_to(root))
            if not include_result and name == _RESULT_NAME:
                continue
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    # -- durable result ---------------------------------------------------
    def publish_result(self, job: Any, result: JobResult) -> str:
        """Persist the small semantic result so the finalizer can be replayed."""
        payload = (canonical_json(result.durable()) + "\n").encode("utf-8")
        atomic_write_bytes(self.artifact_path(job.job_id(), _RESULT_NAME), payload)
        return hashlib.sha256(payload).hexdigest()

    def load_result(self, job: Any) -> JobResult | None:
        path = self.artifact_path(job.job_id(), _RESULT_NAME)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        return JobResult.from_durable(payload)

    # -- resume -----------------------------------------------------------
    def verify_committed(self, job: Any, receipt: Mapping[str, Any]) -> JobState:
        """Validate a receipt already held in memory.

        Separate from :meth:`classify` so the group ledger can resolve resume
        decisions from its index without re-parsing ``receipts.jsonl`` for every
        job. Only this job's own artifact directory is touched.
        """
        if receipt.get("job_identity") != job.identity():
            return JobState(
                STATE_MISMATCH,
                dict(receipt),
                "receipt job identity does not match the planned job",
                phase_id=self.root.name,
            )
        outcome = receipt.get("outcome")
        if outcome not in COMMITTED_OUTCOMES:
            return JobState(
                STATE_PENDING,
                dict(receipt),
                "receipt is not a committed outcome",
                phase_id=self.root.name,
            )
        # Shared committed validation. A terminal reject is as committed as a
        # completed job, so its internal artifacts are validated identically
        # rather than being exempted.
        #
        # ``result.json`` is bound by ``result_digest``, not by ``artifact_digests``.
        # Receipts written before that split also listed it, so it is popped here
        # and checked against the result digest; a legacy receipt that disagrees
        # with itself is still a mismatch. When nothing else is expected, the
        # artifact directory is never walked at all - a purely result-bound job
        # must not pay for a directory scan on resume.
        expected = dict(receipt.get("artifact_digests") or {})
        legacy_result_digest = expected.pop("result.json", None)
        if (
            legacy_result_digest is not None
            and legacy_result_digest != str(receipt.get("result_digest", ""))
        ):
            return JobState(
                STATE_MISMATCH,
                dict(receipt),
                "receipt result digest disagrees with its artifact digest",
                phase_id=self.root.name,
            )
        if expected:
            # Exact set equality over the *binary* artifacts, with ``result.json``
            # excluded from the walk: a tampered, missing or unexpected extra
            # binary artifact is still a mismatch.
            artifacts = self.artifact_digests(job.job_id(), include_result=False)
            if artifacts != expected:
                return JobState(
                    STATE_MISMATCH,
                    dict(receipt),
                    "receipt artifact digests do not match artifacts on disk",
                    phase_id=self.root.name,
                )
        # From here on every committed outcome is validated identically.
        # result.json is mandatory: a receipt without a durable result cannot
        # be replayed, and replaying nothing would silently drop downstream
        # jobs, so it fails closed rather than being treated as legacy state.
        result_path = self.artifact_path(job.job_id(), _RESULT_NAME)
        expected_result = str(receipt.get("result_digest", ""))
        if not expected_result:
            return JobState(
                STATE_MISMATCH,
                dict(receipt),
                "receipt has no durable result digest",
                phase_id=self.root.name,
            )
        if not result_path.is_file():
            return JobState(
                STATE_MISMATCH,
                dict(receipt),
                "durable result is missing",
                phase_id=self.root.name,
            )
        # One read and one digest: the same bytes answer both the digest check
        # and the parse, so the durable result is never read or hashed twice.
        result_bytes = result_path.read_bytes()
        if hashlib.sha256(result_bytes).hexdigest() != expected_result:
            return JobState(
                STATE_MISMATCH,
                dict(receipt),
                "durable result does not match its receipt",
                phase_id=self.root.name,
            )
        try:
            payload = json.loads(result_bytes)
        except ValueError:
            return JobState(
                STATE_MISMATCH,
                dict(receipt),
                "durable result is not parseable",
                phase_id=self.root.name,
            )
        if not isinstance(payload, dict):
            return JobState(
                STATE_MISMATCH,
                dict(receipt),
                "durable result is not an object",
                phase_id=self.root.name,
            )
        if payload.get("outcome") != receipt.get("outcome"):
            return JobState(
                STATE_MISMATCH,
                dict(receipt),
                "durable result outcome disagrees with its receipt",
                phase_id=self.root.name,
            )
        receipt_externals = [
            item for item in receipt.get("external_artifacts", [])
        ]
        if receipt_externals != list(payload.get("external_artifacts", [])):
            return JobState(
                STATE_MISMATCH,
                dict(receipt),
                "durable result external artifacts disagree with the receipt",
                phase_id=self.root.name,
            )
        for item in receipt_externals:
            reference = ArtifactReference.from_record(item)
            if not verify_external_artifact(reference):
                return JobState(
                    STATE_MISMATCH,
                    dict(receipt),
                    f"external artifact failed validation: {reference.path}",
                    phase_id=self.root.name,
                )
        if outcome == OUTCOME_COMPLETED:
            return JobState(STATE_COMPLETED, dict(receipt), phase_id=self.root.name)
        return JobState(
            STATE_TERMINAL_REJECT, dict(receipt), phase_id=self.root.name
        )

    def classify(self, job: Any) -> JobState:
        """Resolve one job's resume state against this phase."""
        receipt = self.receipts().get(job.job_id())
        if receipt is None:
            # One directory scan, not two: the artifact listing answers both the
            # state and its detail.
            artifacts = self.artifact_digests(job.job_id())
            return JobState(
                STATE_RERUN_NO_RECEIPT if artifacts else STATE_PENDING,
                detail="artifact without receipt" if artifacts else "",
                phase_id=self.root.name,
            )
        return self.verify_committed(job, receipt)

    def commit(
        self,
        job: Any,
        *,
        outcome: str,
        artifact_digests: Mapping[str, str],
        result_digest: str = "",
        external_artifacts: tuple[ArtifactReference, ...] = (),
    ) -> Receipt:
        receipt = Receipt(
            job_id=job.job_id(),
            job_identity=job.identity(),
            resource=job.resource,
            outcome=outcome,
            artifact_digests=dict(artifact_digests),
            model_identity=job.model_identity,
            result_digest=result_digest,
            external_artifacts=external_artifacts,
        )
        self.append_receipt(receipt)
        return receipt


class GroupLedger:
    """Aggregate every phase ledger of one resource epoch group.

    A conditional second attempt is planned in a later phase than the job that
    unlocked it, so resume has to locate a receipt without knowing in advance
    which phase produced it.

    A group can contain tens of thousands of model jobs, so the expensive part
    (walking phases and parsing ``receipts.jsonl``) happens **once** during
    :meth:`refresh`. Afterwards :meth:`classify` is an in-memory lookup plus an
    artifact check scoped to that one job. Commits update the index
    incrementally, so a long run never re-scans.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.jsonl_load_calls = 0
        self.refresh_calls = 0
        self._torn_receipts_repaired = 0
        self._receipt_index: dict[str, dict[str, Any]] = {}
        self._job_phase_index: dict[str, str] = {}
        self._artifact_owners: dict[str, str] = {}
        self._dirty = True

    # -- indexing ---------------------------------------------------------
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

    def refresh(self, *, force: bool = False) -> None:
        """Rebuild the in-memory indexes. Cheap when nothing changed."""
        if not self._dirty and not force:
            return
        self.refresh_calls += 1
        receipt_index: dict[str, dict[str, Any]] = {}
        job_phase: dict[str, str] = {}
        artifact_owners: dict[str, str] = {}
        for phase_id in self.phase_ids():
            ledger = self.phase(phase_id)
            records, repaired = ledger.load_receipts()
            self.jsonl_load_calls += 1
            if repaired:
                self._torn_receipts_repaired += 1
            for record in records:
                job_id = record.get("job_id")
                if isinstance(job_id, str):
                    receipt_index[job_id] = record
                    job_phase[job_id] = phase_id
            artifacts_root = ledger.artifacts_root
            if artifacts_root.is_dir():
                for entry in artifacts_root.iterdir():
                    if entry.is_dir():
                        artifact_owners[entry.name] = phase_id
        self._receipt_index = receipt_index
        self._job_phase_index = job_phase
        self._artifact_owners = artifact_owners
        self._dirty = False

    def note_commit(self, phase_id: str, receipt: Receipt) -> None:
        """Keep the index current without re-reading every phase."""
        self._receipt_index[receipt.job_id] = receipt.record()
        self._job_phase_index[receipt.job_id] = phase_id
        self._artifact_owners[receipt.job_id] = phase_id

    def note_artifact(self, phase_id: str, job_id: str) -> None:
        self._artifact_owners[job_id] = phase_id

    # -- lookup -----------------------------------------------------------
    def phase_for(self, job: Any) -> str | None:
        self.refresh()
        return self._job_phase_index.get(job.job_id())

    def classify(self, job: Any) -> JobState:
        self.refresh()
        job_id = job.job_id()
        receipt = self._receipt_index.get(job_id)
        if receipt is None:
            owner = self._artifact_owners.get(job_id)
            # No rebuild here on purpose. During a run the scheduler registers
            # every artifact via note_artifact(), so after refresh() an unknown
            # job genuinely has no artifact yet: it is simply pending. A crash
            # between an artifact rename and note_artifact() also kills the
            # process, so the next restart's refresh() picks it up anyway.
            # Rebuilding per miss would rescan every phase dir for every new
            # pending job, which is O(N^2) on an 80k-row group.
            if owner is None:
                return JobState(STATE_PENDING)
            return JobState(
                STATE_RERUN_NO_RECEIPT,
                detail="artifact without receipt",
                phase_id=owner,
            )
        phase_id = self._job_phase_index.get(job_id) or ""
        return self.phase(phase_id).verify_committed(job, receipt)

    def load_committed_result(self, job: Any) -> JobResult | None:
        """Rebuild the result of an already-committed job for finalizer replay."""
        self.refresh()
        phase_id = self._job_phase_index.get(job.job_id())
        if phase_id is None:
            return None
        return self.phase(phase_id).load_result(job)

    @property
    def torn_receipts_repaired(self) -> int:
        """Repairs observed by *this* ledger, not re-derived per call.

        Recreating a PhaseLedger to read its counter would always report zero,
        so the count is accumulated during refresh.
        """
        return self._torn_receipts_repaired


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
