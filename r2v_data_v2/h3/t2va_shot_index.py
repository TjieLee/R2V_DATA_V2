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
from r2v_data_v2.naming import parse_clip_identity

_OFFSET = struct.Struct("<Q")
_UID_ENTRY = struct.Struct("<12sQ")
_UID_BUCKETS = 16


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


def _uid_index_writer(directory: Path, clips_root: Path):
    """Sort fixed-width UID/row entries in bounded buckets for binary lookup."""
    buckets = [
        (directory / f".uid-bucket-{n:x}").open("wb")
        for n in range(_UID_BUCKETS)
    ]

    def add(line: bytes, index: int) -> None:
        try:
            raw = json.loads(line)
            value = raw.get("video_path") if isinstance(raw, dict) else None
            if not isinstance(value, str) or not value.strip():
                return
            raw_path = Path(value).expanduser()
            video = (
                raw_path if raw_path.is_absolute() else clips_root / raw_path
            ).resolve(strict=False)
            video.relative_to(clips_root)
            uid = bytes.fromhex(parse_clip_identity(video).clip_uid)
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            return
        buckets[uid[0] >> 4].write(_UID_ENTRY.pack(uid, index))

    return buckets, add


def _finish_uid_index(directory: Path, buckets: list, clips_root: Path) -> None:
    for bucket in buckets:
        bucket.close()
    with (directory / "uids.bin").open("wb") as output:
        previous = None
        for n in range(_UID_BUCKETS):
            bucket_path = directory / f".uid-bucket-{n:x}"
            data = bucket_path.read_bytes()
            entries = sorted(
                data[i : i + _UID_ENTRY.size]
                for i in range(0, len(data), _UID_ENTRY.size)
            )
            for entry in entries:
                uid = entry[:12]
                if uid == previous:
                    raise ValueError("duplicate JEA shot clip identity")
                output.write(entry)
                previous = uid
            bucket_path.unlink()
    (directory / "uid-metadata.json").write_text(
        json.dumps({"clips_root": str(clips_root)}, sort_keys=True)
    )


def _ensure_uid_index(path: Path, root: Path, clips_root: Path, identity: dict) -> None:
    try:
        uid_metadata = json.loads((root / "uid-metadata.json").read_text())
        size = (root / "uids.bin").stat().st_size
        if (
            uid_metadata["clips_root"] == str(clips_root)
            and size % _UID_ENTRY.size == 0
        ):
            return
    except (OSError, ValueError, KeyError, TypeError):
        pass
    temporary = root.parent / f".{root.name}.uid-tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        buckets, add = _uid_index_writer(temporary, clips_root)
        try:
            with path.open("rb") as source:
                index = 0
                for line in source:
                    if line.strip():
                        add(line, index)
                        index += 1
        finally:
            for bucket in buckets:
                bucket.close()
        if manifest_identity(path) != identity:
            raise ValueError("JEA shot manifest changed during UID index build")
        _finish_uid_index(temporary, buckets, clips_root)
        os.replace(temporary / "uids.bin", root / "uids.bin")
        os.replace(temporary / "uid-metadata.json", root / "uid-metadata.json")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _index(
    path: Path, cache_root: Path, *, uid_clips_root: Path | None = None
) -> tuple[Path, dict]:
    identity = manifest_identity(path)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = cache_root / key
    if root.exists():
        valid = False
        try:
            metadata = json.loads((root / "metadata.json").read_text())
            offsets = root / "offsets.bin"
            valid = (
                metadata["identity"] == identity
                and metadata["version"] == 1
                and offsets.stat().st_size == metadata["row_count"] * _OFFSET.size
            )
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if valid:
            if uid_clips_root is not None:
                _ensure_uid_index(path, root, uid_clips_root, identity)
            return offsets, metadata
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
        buckets, add = (
            _uid_index_writer(temporary, uid_clips_root)
            if uid_clips_root is not None
            else ([], None)
        )
        try:
            with (
                path.open("rb") as source,
                (temporary / "offsets.bin").open("wb") as output,
            ):
                for line in source:
                    digest.update(line)
                    if line.strip():
                        output.write(_OFFSET.pack(position))
                        if add is not None:
                            add(line, count)
                        count += 1
                    position += len(line)
        finally:
            for bucket in buckets:
                bucket.close()
        if manifest_identity(path) != identity:
            raise ValueError("JEA shot manifest changed during index build")
        if uid_clips_root is not None:
            _finish_uid_index(temporary, buckets, uid_clips_root)
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
            return _index(path, cache_root, uid_clips_root=uid_clips_root)
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


@contextmanager
def indexed_uid_rows(path: Path, cache_root: Path | None, clips_root: Path):
    cache = (
        (cache_root or Path(tempfile.gettempdir()) / "r2v-t2va-shot-index")
        .expanduser()
        .resolve()
    )
    if cache == path.parent or path.parent in cache.parents:
        raise ValueError("shot index must use a separate writable cache directory")
    offsets_path, metadata = _index(path, cache, uid_clips_root=clips_root)
    with (
        path.open("rb") as source,
        offsets_path.open("rb") as offsets,
        (offsets_path.parent / "uids.bin").open("rb") as uids,
    ):
        count = uids.seek(0, os.SEEK_END) // _UID_ENTRY.size

        def lookup(uid: str) -> tuple[int, bytes]:
            try:
                key = bytes.fromhex(uid)
            except ValueError as exc:
                raise ValueError(
                    "selected clips must be unique members of source inventory"
                ) from exc
            if len(key) != 12:
                raise ValueError(
                    "selected clips must be unique members of source inventory"
                )
            left, right = 0, count
            while left < right:
                middle = (left + right) // 2
                uids.seek(middle * _UID_ENTRY.size)
                found, index = _UID_ENTRY.unpack(uids.read(_UID_ENTRY.size))
                if found < key:
                    left = middle + 1
                else:
                    right = middle
            if left == count:
                raise ValueError(
                    "selected clips must be unique members of source inventory"
                )
            uids.seek(left * _UID_ENTRY.size)
            found, index = _UID_ENTRY.unpack(uids.read(_UID_ENTRY.size))
            if found != key:
                raise ValueError(
                    "selected clips must be unique members of source inventory"
                )
            offsets.seek(index * _OFFSET.size)
            source.seek(_OFFSET.unpack(offsets.read(_OFFSET.size))[0])
            return index, source.readline()

        try:
            yield metadata, lookup
        finally:
            if manifest_identity(path) != metadata["identity"]:
                raise ValueError("JEA shot manifest changed during selection")
