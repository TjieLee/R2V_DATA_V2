#!/usr/bin/env python3
"""Prepare immutable fixed-selection shards from growing JEA production JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.v3.config import load_config
from r2v_data_v2.v3.production_source import (
    JEA_VIDEO_MOTION_ADAPTER,
    JeaVideoMotionAdapter,
)

DEFAULT_SOURCE_JSONL = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/shots_f03_motion.jsonl"
)
DEFAULT_STATE_ROOT = Path(
    "/mnt/workspace/litengjie/data/r2v_v3_configs/production/"
    "jea_motion_v1/prod-v1"
)
DEFAULT_SHARD_SIZE = 1000
PRODUCTION_SOURCE_SCHEMA_VERSION = "r2v.v3.production-source.1"


def _below(path: Path, root: Path) -> bool:
    return root in path.parents


def _resolve_source_jsonl(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=True)
    dataset_root = config_module.ALLOWED_DATASET_ROOT.resolve(strict=False)
    if (
        not resolved.is_file()
        or resolved.suffix.lower() != ".jsonl"
        or not _below(resolved, dataset_root)
    ):
        raise ValueError("source_jsonl must be a JSONL below public dataset")
    return resolved


def _resolve_state_root(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve(strict=False)
    config_root = (
        config_module.ALLOWED_WRITABLE_ROOT / "r2v_v3_configs"
    ).resolve(strict=False)
    if not _below(resolved, config_root):
        raise ValueError("state_root must be below r2v_v3_configs")
    return resolved


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise FileExistsError(f"immutable production artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if not path.is_file() or path.read_bytes() != content:
                raise FileExistsError(
                    f"immutable production artifact differs: {path}"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _base_config_identity(path: Path) -> tuple[bytes, str, str]:
    content = path.read_bytes()
    fingerprint = load_config(path).fingerprint()
    if path.read_bytes() != content:
        raise ValueError("base config changed while its identity was computed")
    return content, _sha256_bytes(content), fingerprint


def _load_cursor(path: Path, source_jsonl: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "source_jsonl": str(source_jsonl),
            "adapter_version": JEA_VIDEO_MOTION_ADAPTER,
            "next_source_index": 0,
            "next_line_number": 1,
            "next_byte_offset": 0,
            "last_committed_line_start_offset": None,
            "last_committed_line_sha256": None,
        }
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("cursor.json must contain an object")
    if value.get("source_jsonl") != str(source_jsonl):
        raise ValueError("cursor source_jsonl does not match")
    if value.get("adapter_version") != JEA_VIDEO_MOTION_ADAPTER:
        raise ValueError("cursor adapter_version does not match")
    for field_name in (
        "next_source_index",
        "next_line_number",
        "next_byte_offset",
    ):
        field_value = value.get(field_name)
        minimum = 1 if field_name == "next_line_number" else 0
        if (
            not isinstance(field_value, int)
            or isinstance(field_value, bool)
            or field_value < minimum
        ):
            raise ValueError(f"cursor {field_name} is invalid")
    return value


def _validate_committed_prefix(source_jsonl: Path, cursor: dict[str, object]) -> None:
    next_offset = int(cursor["next_byte_offset"])
    if source_jsonl.stat().st_size < next_offset:
        raise ValueError("production source was truncated before committed offset")
    line_start = cursor.get("last_committed_line_start_offset")
    expected_sha = cursor.get("last_committed_line_sha256")
    if line_start is None and expected_sha is None and next_offset == 0:
        return
    if (
        not isinstance(line_start, int)
        or line_start < 0
        or not isinstance(expected_sha, str)
    ):
        raise ValueError("cursor committed-line identity is invalid")
    with source_jsonl.open("rb") as handle:
        handle.seek(line_start)
        line = handle.readline()
        boundary_end = handle.tell()
        if boundary_end > next_offset:
            raise ValueError("cursor committed-line offsets are inconsistent")
        consumed_after_boundary = handle.read(next_offset - boundary_end)
    if consumed_after_boundary.strip():
        raise ValueError("cursor committed-line offsets are inconsistent")
    if _sha256_bytes(line) != expected_sha:
        raise ValueError("production source committed prefix changed")


def _probe_source_paths(
    source_jsonl: Path,
    adapter: JeaVideoMotionAdapter,
    *,
    required_records: int,
) -> dict[str, int]:
    if required_records < 1:
        raise ValueError("path_probe_records must be positive")
    inspected = 0
    examined = 0
    maximum_examined = required_records * 10
    source_video_limit = min(required_records, 100)
    source_videos: set[str] = set()
    with source_jsonl.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            examined += 1
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("production source record must be an object")
                clip_path, _ = adapter.resolve_clip_path(value)
                if clip_path.suffix.lower() != ".mp4":
                    raise ValueError("record.video_path must identify an MP4")
                source_value = value.get("source_video_path")
                if (
                    len(source_videos) < source_video_limit
                    and isinstance(source_value, str)
                    and source_value not in source_videos
                ):
                    source_path, _ = adapter.resolve_source_video_path(
                        value,
                        require_file=True,
                    )
                    if source_path.suffix.lower() != ".mp4":
                        raise ValueError(
                            "record.source_video_path must identify an MP4"
                        )
                    source_videos.add(source_value)
            except Exception:  # noqa: BLE001 - path discovery skips malformed rows
                if examined >= maximum_examined:
                    break
                continue
            inspected += 1
            if inspected >= required_records and source_videos:
                return {
                    "clip_paths_verified": inspected,
                    "unique_source_video_paths_verified": len(source_videos),
                }
            if examined >= maximum_examined:
                break
    raise ValueError(
        f"fewer than {required_records} of the first {examined} production records "
        "resolved below clips_root with an existing source video"
    )


def _source_descriptor(
    *,
    source_jsonl: Path,
    base_config: Path,
    adapter: JeaVideoMotionAdapter,
    shard_size: int,
    path_probe_records: int,
    base_config_sha256: str,
    base_config_fingerprint: str,
) -> bytes:
    value = {
        "schema_version": PRODUCTION_SOURCE_SCHEMA_VERSION,
        "source_adapter": JEA_VIDEO_MOTION_ADAPTER,
        "source_jsonl": str(source_jsonl),
        "base_config_path": str(base_config),
        "base_config_sha256": base_config_sha256,
        "base_config_fingerprint": base_config_fingerprint,
        "clips_root": str(adapter.clips_root),
        "source_videos_root": str(adapter.source_videos_root),
        "path_probe_records": path_probe_records,
        "shard_size": shard_size,
    }
    return yaml.safe_dump(value, sort_keys=False).encode("utf-8")


def _validate_source_descriptor(
    path: Path,
    *,
    source_jsonl: Path,
    base_config: Path,
    base_config_sha256: str,
    base_config_fingerprint: str,
    adapter: JeaVideoMotionAdapter,
    shard_size: int,
) -> None:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("source.yaml must contain a YAML mapping")
    expected = {
        "schema_version": PRODUCTION_SOURCE_SCHEMA_VERSION,
        "source_adapter": JEA_VIDEO_MOTION_ADAPTER,
        "source_jsonl": str(source_jsonl),
        "base_config_path": str(base_config),
        "base_config_sha256": base_config_sha256,
        "base_config_fingerprint": base_config_fingerprint,
        "clips_root": str(adapter.clips_root),
        "source_videos_root": str(adapter.source_videos_root),
        "shard_size": shard_size,
    }
    for field_name, expected_value in expected.items():
        if value.get(field_name) != expected_value:
            raise ValueError(
                f"immutable source.yaml field differs: {field_name}"
            )


def _selection_bytes(records: list[dict[str, object]]) -> bytes:
    return b"".join(_json_bytes(record) for record in records)


def _shard_config_bytes(
    base: dict[str, object],
    *,
    source_jsonl: Path,
    selection_path: Path,
    shard_id: str,
) -> bytes:
    value = dict(base)
    source = dict(value.get("source") or {})
    source.update(
        {
            "selection_mode": "fixed_selection_v1",
            "selection_manifest": str(selection_path),
            "start_index": 0,
            "limit": None,
            "allow_full_run": False,
        }
    )
    writable = config_module.ALLOWED_WRITABLE_ROOT.resolve(strict=False)
    value.update(
        {
            "dataset_json": str(source_jsonl),
            "run_root": str(
                writable
                / "r2v_v3_runs"
                / "production"
                / "jea_motion_v1"
                / "prod-v1"
                / shard_id
            ),
            "export_root": str(
                writable
                / "r2v_v3_exports"
                / "production"
                / "jea_motion_v1"
                / "prod-v1"
                / "shards"
                / shard_id
            ),
            "source": source,
        }
    )
    return yaml.safe_dump(value, sort_keys=False).encode("utf-8")


def _error_record(
    *,
    line_number: int,
    source_index: int,
    reason: str,
    raw: object,
) -> dict[str, object]:
    result: dict[str, object] = {
        "line_number": line_number,
        "source_index": source_index,
        "reason": reason[:2048],
    }
    if isinstance(raw, dict):
        for field_name in ("video_path", "source_video_path"):
            value = raw.get(field_name)
            if isinstance(value, str):
                result[field_name] = value[:2048]
    return result


def _append_errors(path: Path, errors: list[dict[str, object]]) -> None:
    if not errors:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for error in errors:
            handle.write(_json_bytes(error))
        handle.flush()
        os.fsync(handle.fileno())


def probe_production_setup(
    *,
    source_jsonl: str | Path,
    base_config: str | Path,
    clips_root: str | Path,
    source_videos_root: str | Path,
    path_probe_records: int = 100,
) -> dict[str, object]:
    source_path = _resolve_source_jsonl(source_jsonl)
    base_path = Path(base_config).expanduser().resolve(strict=True)
    _, base_sha256, base_fingerprint = _base_config_identity(base_path)
    adapter = JeaVideoMotionAdapter.create(
        clips_root=clips_root,
        source_videos_root=source_videos_root,
    )
    counts = _probe_source_paths(
        source_path,
        adapter,
        required_records=path_probe_records,
    )
    return {
        "probe_only": True,
        "source_jsonl": str(source_path),
        "base_config_path": str(base_path),
        "base_config_sha256": base_sha256,
        "base_config_fingerprint": base_fingerprint,
        "clips_root": str(adapter.clips_root),
        "source_videos_root": str(adapter.source_videos_root),
        **counts,
    }


def prepare_production_shards(
    *,
    source_jsonl: str | Path,
    base_config: str | Path,
    clips_root: str | Path,
    source_videos_root: str | Path,
    state_root: str | Path = DEFAULT_STATE_ROOT,
    shard_size: int = DEFAULT_SHARD_SIZE,
    seal_tail: bool = False,
    path_probe_records: int = 100,
    max_shards: int | None = None,
) -> list[Path]:
    if not isinstance(shard_size, int) or isinstance(shard_size, bool) or shard_size < 1:
        raise ValueError("shard_size must be a positive integer")
    if max_shards is not None and (
        not isinstance(max_shards, int)
        or isinstance(max_shards, bool)
        or max_shards < 1
    ):
        raise ValueError("max_shards must be a positive integer")
    source_path = _resolve_source_jsonl(source_jsonl)
    production_root = _resolve_state_root(state_root)
    base_path = Path(base_config).expanduser().resolve(strict=True)
    base_bytes, base_config_sha256, base_config_fingerprint = (
        _base_config_identity(base_path)
    )
    adapter = JeaVideoMotionAdapter.create(
        clips_root=clips_root,
        source_videos_root=source_videos_root,
    )
    source_yaml = production_root / "source.yaml"
    if source_yaml.exists():
        _validate_source_descriptor(
            source_yaml,
            source_jsonl=source_path,
            base_config=base_path,
            base_config_sha256=base_config_sha256,
            base_config_fingerprint=base_config_fingerprint,
            adapter=adapter,
            shard_size=shard_size,
        )
    else:
        _probe_source_paths(
            source_path,
            adapter,
            required_records=path_probe_records,
        )
        descriptor = _source_descriptor(
            source_jsonl=source_path,
            base_config=base_path,
            adapter=adapter,
            shard_size=shard_size,
            path_probe_records=path_probe_records,
            base_config_sha256=base_config_sha256,
            base_config_fingerprint=base_config_fingerprint,
        )
        _write_immutable(source_yaml, descriptor)

    base = yaml.safe_load(base_bytes.decode("utf-8"))
    if not isinstance(base, dict):
        raise TypeError("base V3 config must be a YAML mapping")
    cursor_path = production_root / "state" / "cursor.json"
    cursor = _load_cursor(cursor_path, source_path)
    _validate_committed_prefix(source_path, cursor)
    if not cursor_path.exists():
        _atomic_write(cursor_path, _json_bytes(cursor))

    selections = production_root / "selections"
    shards = production_root / "shards"
    errors_path = production_root / "state" / "source_errors.jsonl"
    sealed: list[Path] = []
    next_index = int(cursor["next_source_index"])
    next_line_number = int(cursor["next_line_number"])
    next_offset = int(cursor["next_byte_offset"])
    pending_records: list[dict[str, object]] = []
    pending_clip_uids: set[str] = set()
    pending_errors: list[dict[str, object]] = []
    pending_count = 0
    shard_start = next_index
    last_line_start: int | None = None
    last_line_sha: str | None = None

    def seal(end_index: int, end_offset: int) -> None:
        nonlocal pending_count, shard_start, pending_records, pending_errors
        nonlocal pending_clip_uids
        shard_id = f"shard-{shard_start:09d}-{end_index:09d}"
        selection_path = selections / f"{shard_id}.jsonl"
        shard_config = shards / f"{shard_id}.yaml"
        _write_immutable(selection_path, _selection_bytes(pending_records))
        _write_immutable(
            shard_config,
            _shard_config_bytes(
                base,
                source_jsonl=source_path,
                selection_path=selection_path.resolve(strict=False),
                shard_id=shard_id,
            ),
        )
        load_config(shard_config)
        _append_errors(errors_path, pending_errors)
        assert last_line_start is not None and last_line_sha is not None
        committed = {
            "source_jsonl": str(source_path),
            "adapter_version": JEA_VIDEO_MOTION_ADAPTER,
            "next_source_index": end_index + 1,
            "next_line_number": next_line_number,
            "next_byte_offset": end_offset,
            "last_committed_line_start_offset": last_line_start,
            "last_committed_line_sha256": last_line_sha,
        }
        _atomic_write(cursor_path, _json_bytes(committed))
        sealed.append(shard_config)
        shard_start = end_index + 1
        pending_count = 0
        pending_records = []
        pending_clip_uids = set()
        pending_errors = []

    with source_path.open("rb") as handle:
        handle.seek(next_offset)
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number = next_line_number
            next_line_number += 1
            if not line.strip():
                continue
            source_index = next_index
            next_index += 1
            pending_count += 1
            last_line_start = line_start
            last_line_sha = hashlib.sha256(line).hexdigest()
            raw: object = None
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise TypeError("production source record must be a JSON object")
                selected = adapter.parse(raw, source_index=source_index)
                clip_uid = str(selected["clip_uid"])
                if clip_uid in pending_clip_uids:
                    raise ValueError(f"duplicate clip_uid in shard: {clip_uid}")
                _json_bytes(selected)
                pending_clip_uids.add(clip_uid)
                pending_records.append(selected)
            except Exception as exc:  # noqa: BLE001 - isolate one source line
                pending_errors.append(
                    _error_record(
                        line_number=line_number,
                        source_index=source_index,
                        reason=str(exc),
                        raw=raw,
                    )
                )
            if pending_count == shard_size:
                seal(source_index, handle.tell())
                if max_shards is not None and len(sealed) >= max_shards:
                    return sealed
        if pending_count and seal_tail:
            seal(next_index - 1, handle.tell())
    return sealed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--source-videos-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--path-probe-records", type=int, default=100)
    parser.add_argument("--seal-tail", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--max-shards", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.probe_only:
        result = probe_production_setup(
            source_jsonl=args.source_jsonl,
            base_config=args.base_config,
            clips_root=args.clips_root,
            source_videos_root=args.source_videos_root,
            path_probe_records=args.path_probe_records,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return
    paths = prepare_production_shards(
        source_jsonl=args.source_jsonl,
        base_config=args.base_config,
        clips_root=args.clips_root,
        source_videos_root=args.source_videos_root,
        state_root=args.state_root,
        shard_size=args.shard_size,
        seal_tail=args.seal_tail,
        path_probe_records=args.path_probe_records,
        max_shards=args.max_shards,
    )
    print(json.dumps({"sealed_shards": [str(path) for path in paths]}))


if __name__ == "__main__":
    main()
