"""Target-side production orchestration; frozen semantic modules remain unchanged."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import shutil
import threading
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

SHARD_SIZE = 10_000
DEFAULT_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA"
)


def shard_bounds(shard_id: int) -> tuple[int, int]:
    if shard_id < 0:
        raise ValueError("negative shard")
    start = shard_id * SHARD_SIZE
    return start, start + SHARD_SIZE - 1


def shard_name(shard_id: int) -> str:
    start, end = shard_bounds(shard_id)
    return f"shard-{start:09d}-{end:09d}"


def shard_order(*, explicit=None, start=0, end=0, seed=20260917, rank=0, world_size=1):
    if explicit is not None:
        result = list(explicit)
        if not result or len(set(result)) != len(result) or min(result) < 0:
            raise ValueError("shards must be unique nonnegative IDs")
        return result
    if start < 0 or end < start or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid shard schedule")
    result = list(range(start, end + 1))
    random.Random(seed).shuffle(result)
    return result[rank::world_size]


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class ShardLockedError(RuntimeError):
    pass


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ShardLockedError(f"locked: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _source_identity(path):
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def build_source_index(source: Path, root: Path) -> dict:
    source = source.resolve(strict=True)
    root = root.resolve()
    if source.is_relative_to(root):
        raise ValueError("source must not be inside production output")
    path = root / "manifests/source_index.json"
    with file_lock(root / "manifests/source_index.lock"):
        identity = _source_identity(source)
        if path.exists():
            index = json.loads(path.read_text())
            if (
                index["source_identity"] != identity
                or index["shard_size"] != SHARD_SIZE
            ):
                raise ValueError("source changed; use a fresh production root")
            return index
        digest = hashlib.sha256()
        shards, count = [], 0
        with source.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if count % SHARD_SIZE == 0:
                    shards.append(
                        {
                            "shard_id": count // SHARD_SIZE,
                            "start_index": count,
                            "end_index": count,
                            "byte_offset": offset,
                        }
                    )
                shards[-1]["end_index"] = count
                count += 1
                digest.update(line)
        if _source_identity(source) != identity:
            raise ValueError("source changed during indexing")
        index = {
            "shard_size": SHARD_SIZE,
            "source_record_count": count,
            "source_identity": identity,
            "source_sha256": digest.hexdigest(),
            "shards": shards,
        }
        atomic_json(path, index)
        return index


def materialize_shard(index: dict, shard_id: int, root: Path) -> Path:
    source = Path(index["source_identity"]["path"])
    if _source_identity(source) != index["source_identity"]:
        raise ValueError("source changed after indexing")
    if not 0 <= shard_id < len(index["shards"]):
        raise ValueError("shard outside source population")
    shard = index["shards"][shard_id]
    path = root / "manifests" / f"{shard_name(shard_id)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            return path
        temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            with source.open("rb") as handle, temporary.open("wb") as output:
                handle.seek(shard["byte_offset"])
                for _ in range(shard["end_index"] - shard["start_index"] + 1):
                    output.write(handle.readline())
                output.flush()
                os.fsync(output.fileno())
            if _source_identity(source) != index["source_identity"]:
                raise ValueError("source changed during shard extraction")
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)
    return path


class EndpointUnavailable(RuntimeError):
    """Infrastructure outage: stop scheduling, do not label remaining samples."""


def complete_rows(path: Path):
    if not path.exists():
        return
    with path.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def _repair_tail(path: Path):
    if not path.exists():
        return
    with path.open("r+b") as handle:
        end = 0
        for line in handle:
            if not line.endswith(b"\n"):
                handle.truncate(end)
                handle.flush()
                os.fsync(handle.fileno())
                return
            end = handle.tell()


def append_row(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_states(shard: Path) -> dict:
    return {r["clip_uid"]: r for r in complete_rows(shard / "state.jsonl.partial")}


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _recover_stage(destination: Path, identity: str):
    if not destination.exists():
        return None
    receipt = json.loads((destination / "stage.json").read_text())
    if receipt["identity"] != identity:
        raise ValueError("production stage identity changed; use a fresh root")
    for relative, digest in receipt["files"].items():
        if _sha(destination / relative) != digest:
            raise ValueError("published stage artifact changed")
    return receipt["result"]


def _publish_stage(temporary: Path, destination: Path, identity: str, result):
    files = {}
    for path in temporary.rglob("*"):
        if path.is_file():
            files[str(path.relative_to(temporary))] = _sha(path)
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    atomic_json(
        temporary / "stage.json",
        {
            "identity": identity,
            "files": files,
            "result": result,
        },
    )
    if destination.exists():
        raise FileExistsError(destination)
    temporary.rename(destination)
    _sync_directory(destination.parent)


def process_shard(
    root: Path,
    shard_id: int,
    rows: list[dict],
    processor,
    *,
    request_workers=8,
    progress_every=100,
):
    """Processor owns semantics; this function owns durable scheduling only."""
    if request_workers < 1:
        raise ValueError("request_workers must be positive")
    shard = root / "shards" / shard_name(shard_id)
    with file_lock(shard / "shard.lock"):
        journal = shard / "state.jsonl.partial"
        _repair_tail(journal)
        states = load_states(shard)
        writer = threading.Lock()
        stopped = threading.Event()
        processed = 0

        def save(state):
            with writer:
                append_row(journal, state)
                states[state["clip_uid"]] = dict(state)

        def process(row):
            nonlocal processed
            uid = row["clip_uid"]
            if Path(uid).name != uid or uid in {".", ".."}:
                raise ValueError("unsafe clip identity")
            identity = processor.identity(row)
            state = dict(
                states.get(uid)
                or {
                    "source_index": row["source_index"],
                    "clip_uid": uid,
                    "t2va_status": "pending",
                    "ta2va_status": "pending",
                    "failure_stage": None,
                    "failure_reason": None,
                    "identity": identity,
                }
            )
            if state["identity"] != identity:
                raise ValueError("production source/config identity changed")
            if state["t2va_status"] == "skipped":
                return
            if row.get("upstream_failure"):
                state.update(
                    t2va_status="skipped",
                    ta2va_status="skipped",
                    failure_reason=row["upstream_failure"],
                    failure_stage="upstream",
                )
                save(state)
                return
            for stage in ("t2va", "ta2va"):
                if stopped.is_set():
                    return
                destination = shard / "artifacts" / uid / stage
                result = _recover_stage(destination, identity)
                if result is None:
                    temporary = destination.with_name(
                        f".{stage}.tmp-{uuid.uuid4().hex}"
                    )
                    temporary.mkdir(parents=True)
                    try:
                        try:
                            result = processor.process(
                                stage, row, temporary, destination
                            )
                        except EndpointUnavailable:
                            stopped.set()
                            raise
                        except Exception as exc:  # noqa: BLE001 - sample isolation boundary.
                            state.update(
                                {
                                    f"{stage}_status": "failed",
                                    "failure_stage": stage,
                                    "failure_reason": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            if stage == "t2va":
                                state["ta2va_status"] = "skipped"
                            save(state)
                            return
                        _publish_stage(temporary, destination, identity, result)
                    finally:
                        if temporary.exists():
                            shutil.rmtree(temporary)
                state.update(
                    {
                        f"{stage}_status": "ready",
                        "failure_stage": None,
                        "failure_reason": None,
                    }
                )
                save(state)
            with writer:
                processed += 1
                if progress_every and processed % progress_every == 0:
                    counts = {
                        f"{s}_{v}": sum(x[f"{s}_status"] == v for x in states.values())
                        for s in ("t2va", "ta2va")
                        for v in ("ready", "failed")
                    }
                    print(
                        f"shard={shard_id} processed={processed}/{len(rows)} {counts}",
                        flush=True,
                    )

        pool = ThreadPoolExecutor(max_workers=request_workers)
        pending = set()
        iterator = iter(rows)
        try:
            while not stopped.is_set():
                while len(pending) < request_workers:
                    row = next(iterator, None)
                    if row is None:
                        break
                    pending.add(pool.submit(process, row))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
        except BaseException:
            stopped.set()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        if stopped.is_set():
            raise EndpointUnavailable("endpoint unavailable; resume after recovery")
        return states
