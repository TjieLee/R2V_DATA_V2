"""Target-side production orchestration; frozen semantic modules remain unchanged."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import uuid
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
