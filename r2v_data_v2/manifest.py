from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import ijson

from r2v_data_v2.config import PipelineConfig
from r2v_data_v2.naming import parse_clip_identity


@dataclass(frozen=True)
class ManifestBuildStats:
    processed: int = 0
    skipped_existing: int = 0
    missing_videos: int = 0


def _first_non_whitespace(path: Path) -> str:
    with path.open("rb") as handle:
        while chunk := handle.read(4096):
            stripped = chunk.lstrip()
            if stripped:
                return chr(stripped[0])
    return ""


def _iter_ijson(path: Path, prefix: str) -> Iterator[dict[str, Any]]:
    with path.open("rb") as handle:
        for value in ijson.items(handle, prefix):
            if not isinstance(value, dict):
                raise TypeError("source manifest records must be JSON objects")
            yield value


def iter_source_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(
                        f"source manifest line {line_number} must be a JSON object"
                    )
                yield value
        return
    if source.suffix.lower() != ".json":
        raise ValueError("dataset_json must use .json or .jsonl")
    first = _first_non_whitespace(source)
    if first == "[":
        yield from _iter_ijson(source, "item")
        return
    if first != "{":
        raise ValueError("source JSON must be an array or object")
    for prefix in ("records.item", "data.item"):
        found = False
        for value in _iter_ijson(source, prefix):
            found = True
            yield value
        if found:
            return
    raise ValueError("source JSON object must contain a records or data array")


def _record_video_path(raw: dict[str, Any]) -> Path:
    for key in ("video_path", "source_video_path", "file_path", "video"):
        value = raw.get(key)
        if isinstance(value, str) and value:
            return Path(value).expanduser().resolve(strict=False)
    raise ValueError("source record has no video path")


def _record_caption(raw: dict[str, Any]) -> str:
    for key in ("caption_raw", "caption", "text"):
        value = raw.get(key)
        if isinstance(value, str):
            return value
    return ""


def _existing_clip_uids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                result.add(str(value["clip_uid"]))
    return result


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def build_manifest(
    config: PipelineConfig,
    *,
    limit: int | None = None,
    start_index: int = 0,
    overwrite: bool = False,
) -> ManifestBuildStats:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    output_root = config.ensure_output_root()
    destination = output_root / "manifests" / "source.jsonl"
    missing_log = output_root / "logs" / "missing_videos.jsonl"
    if overwrite:
        destination.unlink(missing_ok=True)
    existing = _existing_clip_uids(destination)

    processed = skipped = missing = selected = 0
    for source_index, raw in enumerate(iter_source_records(config.dataset_json)):
        if source_index < start_index:
            continue
        if limit is not None and selected >= limit:
            break
        selected += 1
        try:
            video_path = _record_video_path(raw)
            identity = parse_clip_identity(video_path)
        except ValueError as exc:
            _append_jsonl(
                missing_log,
                {
                    "source_index": source_index,
                    "reason": str(exc),
                    "record": raw,
                },
            )
            missing += 1
            continue
        if not video_path.is_file():
            _append_jsonl(
                missing_log,
                {
                    "source_index": source_index,
                    "video_path": str(video_path),
                    "reason": "video file is missing",
                },
            )
            missing += 1
            continue
        if identity.clip_uid in existing:
            skipped += 1
            continue
        record = {
            "source_index": source_index,
            "clip_uid": identity.clip_uid,
            "video_path": str(video_path),
            "caption_raw": _record_caption(raw),
            "parent_video_id": identity.parent_video_id,
            "clip_suffix": identity.clip_suffix,
            "clip_order": list(identity.clip_order),
        }
        _append_jsonl(destination, record)
        existing.add(identity.clip_uid)
        processed += 1
    return ManifestBuildStats(
        processed=processed,
        skipped_existing=skipped,
        missing_videos=missing,
    )


def stats_dict(stats: ManifestBuildStats) -> dict[str, int]:
    return asdict(stats)
