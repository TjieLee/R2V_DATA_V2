"""Disk-backed JSONL row offsets; no shot parsing or source-directory writes."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import random
import shutil
import struct
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from r2v_data_v2.h3.sam_audio_stem_shadow import sha256_file

_OFFSET = struct.Struct("<Q")


def manifest_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _index(path: Path, cache_root: Path) -> tuple[Path, dict]:
    identity = manifest_identity(path)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = cache_root / key
    if root.exists():
        try:
            metadata = json.loads((root / "metadata.json").read_text())
            offsets = root / "offsets.bin"
            if (
                metadata["identity"] == identity
                and metadata["version"] == 1
                and offsets.stat().st_size == metadata["row_count"] * _OFFSET.size
                and sha256_file(offsets) == metadata["offsets_sha256"]
            ):
                return offsets, metadata
        except (OSError, ValueError, KeyError, TypeError):
            pass
        # Quarantine only this cache entry, never the manifest or its directory.
        damaged = cache_root / f".{key}.invalid-{uuid.uuid4().hex}"
        try:
            root.rename(damaged)
        except FileNotFoundError:
            pass
        else:
            if damaged.is_symlink() or damaged.is_file():
                damaged.unlink()
            else:
                shutil.rmtree(damaged)
    temporary = cache_root / f".{key}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        digest = hashlib.sha256()
        count, position = 0, 0
        with (
            path.open("rb") as source,
            (temporary / "offsets.bin").open("wb") as output,
        ):
            for line in source:
                digest.update(line)
                if line.strip():
                    output.write(_OFFSET.pack(position))
                    count += 1
                position += len(line)
        if manifest_identity(path) != identity:
            raise ValueError("JEA shot manifest changed during index build")
        metadata = {
            "version": 1,
            "identity": identity,
            "row_count": count,
            "manifest_sha256": digest.hexdigest(),
            "offsets_sha256": sha256_file(temporary / "offsets.bin"),
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))
        try:
            os.rename(temporary, root)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # Another reader may have published the same immutable index.
            return _index(path, cache_root)
        return root / "offsets.bin", metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def random_indices(count: int, seed: int):
    """Lazy Fisher-Yates permutation, O(draws) state instead of O(population)."""
    rng = random.Random(seed)
    swaps = {}
    for remaining in range(count, 0, -1):
        pick = rng.randrange(remaining)
        value = swaps.get(pick, pick)
        last = swaps.pop(remaining - 1, remaining - 1)
        if pick != remaining - 1:
            swaps[pick] = last
        yield value


@contextmanager
def indexed_rows(path: Path, cache_root: Path | None):
    cache = (
        (cache_root or Path(tempfile.gettempdir()) / "r2v-t2va-shot-index")
        .expanduser()
        .resolve()
    )
    if cache == path.parent or path.parent in cache.parents:
        raise ValueError("shot index must use a separate writable cache directory")
    offsets_path, metadata = _index(path, cache)
    with path.open("rb") as source, offsets_path.open("rb") as offsets:

        def read(index):
            if not 0 <= index < metadata["row_count"]:
                raise ValueError("shot row index outside manifest")
            offsets.seek(index * _OFFSET.size)
            source.seek(_OFFSET.unpack(offsets.read(_OFFSET.size))[0])
            return source.readline()

        try:
            yield metadata, read
        finally:
            if manifest_identity(path) != metadata["identity"]:
                raise ValueError("JEA shot manifest changed during selection")
