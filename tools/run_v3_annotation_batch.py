#!/usr/bin/env python3
"""Run standalone, restart-safe V3 annotation shards for the JEA dataset.

This utility intentionally keeps restart state local to its annotation shards.
Repository commits are informationally irrelevant to shard restart identity.

Inspect one completed part::

    python tools/run_v3_annotation_batch.py --inspect /path/to/shard.jsonl

Plain Python readers need no V3 storage objects::

    for line in open(part_path, encoding="utf-8"):
        row = json.loads(line)
        for entity in row["entities"]:
            print(entity["entity_id"], entity["phrase"])
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import re
import sys
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.annotation import (
    AnnotationClient,
    AnnotationFailure,
    QwenAnnotationClient,
    sanitize_annotation_payload,
)
from r2v_data_v2.v3.config import QwenAnnotationConfig, QwenVideoConfig
from r2v_data_v2.v3.production_source import JeaVideoMotionAdapter
from r2v_data_v2.v3.schemas import AnnotationState

DEFAULT_SOURCE_JSONL = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/shots_f03_motion.jsonl"
)
DEFAULT_CLIPS_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/clips_clean_cropped"
)
DEFAULT_SOURCE_VIDEOS_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/moive-183t-0808"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/entity_annotations"
)
DEFAULT_SHARD_SIZE = 10_000
_WORKER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

# Re-exported deliberately: the batch path uses the existing annotation client,
# whose annotate() method calls this exact sanitizer.
ANNOTATION_SANITIZER = sanitize_annotation_payload


@dataclass(frozen=True)
class AnnotationBatchWorker:
    name: str
    base_url: str
    shard_start: int
    shard_end: int


@dataclass(frozen=True)
class AnnotationBatchSettings:
    source_jsonl: Path
    clips_root: Path
    source_videos_root: Path
    output_root: Path
    shard_size: int
    annotation_model: str
    api_key: str
    temperature: float
    max_tokens: int
    timeout_seconds: int
    repair_retries: int
    entity_selection_mode: str
    fps: float
    workers: tuple[AnnotationBatchWorker, ...]


@dataclass(frozen=True)
class SourceRow:
    source_index: int
    raw: dict[str, object] | None
    read_error: str | None = None


@dataclass(frozen=True)
class ShardResult:
    shard_id: int
    path: Path
    rows: int
    ready: int
    failed: int
    skipped: bool


class ShardLockedError(RuntimeError):
    pass


def shard_bounds(shard_id: int, shard_size: int = DEFAULT_SHARD_SIZE) -> tuple[int, int]:
    if not isinstance(shard_id, int) or isinstance(shard_id, bool) or shard_id < 0:
        raise ValueError("shard_id must be a non-negative integer")
    if (
        not isinstance(shard_size, int)
        or isinstance(shard_size, bool)
        or shard_size < 1
    ):
        raise ValueError("shard_size must be a positive integer")
    start = shard_id * shard_size
    return start, start + shard_size - 1


def select_worker(
    settings: AnnotationBatchSettings,
    name: str,
    *,
    base_url: str | None = None,
    shard_start: int | None = None,
    shard_end: int | None = None,
) -> AnnotationBatchWorker:
    matches = [worker for worker in settings.workers if worker.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown annotation batch worker: {name}")
    selected = replace(
        matches[0],
        base_url=base_url if base_url is not None else matches[0].base_url,
        shard_start=(
            shard_start if shard_start is not None else matches[0].shard_start
        ),
        shard_end=shard_end if shard_end is not None else matches[0].shard_end,
    )
    _validate_worker(selected)
    return selected


def _integer(value: object, *, field_name: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _number(value: object, *, field_name: str, minimum: float) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    result = float(value)
    if result < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return result


def _nonempty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_worker(worker: AnnotationBatchWorker) -> None:
    if _WORKER_NAME.fullmatch(worker.name) is None:
        raise ValueError("worker name contains unsupported characters")
    _nonempty_string(worker.base_url, field_name=f"workers.{worker.name}.base_url")
    _integer(worker.shard_start, field_name="worker.shard_start", minimum=0)
    _integer(worker.shard_end, field_name="worker.shard_end", minimum=0)
    if worker.shard_end < worker.shard_start:
        raise ValueError("worker.shard_end must be >= worker.shard_start")


def load_batch_settings(path: str | Path) -> AnnotationBatchSettings:
    resolved = Path(path).expanduser().resolve(strict=True)
    value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("annotation batch config must contain a YAML mapping")
    annotation = value.get("annotation", {})
    if not isinstance(annotation, dict):
        raise TypeError("annotation config must contain a mapping")
    raw_workers = value.get("workers")
    if not isinstance(raw_workers, list) or not raw_workers:
        raise ValueError("annotation batch config requires at least one worker")
    workers: list[AnnotationBatchWorker] = []
    for index, raw_worker in enumerate(raw_workers):
        if not isinstance(raw_worker, dict):
            raise TypeError(f"workers[{index}] must contain a mapping")
        worker = AnnotationBatchWorker(
            name=_nonempty_string(
                raw_worker.get("name"), field_name=f"workers[{index}].name"
            ),
            base_url=_nonempty_string(
                raw_worker.get("base_url"),
                field_name=f"workers[{index}].base_url",
            ),
            shard_start=_integer(
                raw_worker.get("shard_start"),
                field_name=f"workers[{index}].shard_start",
                minimum=0,
            ),
            shard_end=_integer(
                raw_worker.get("shard_end"),
                field_name=f"workers[{index}].shard_end",
                minimum=0,
            ),
        )
        _validate_worker(worker)
        workers.append(worker)
    names = [worker.name for worker in workers]
    if len(set(names)) != len(names):
        raise ValueError("annotation batch worker names must be unique")

    settings = AnnotationBatchSettings(
        source_jsonl=Path(value.get("source_jsonl", DEFAULT_SOURCE_JSONL)),
        clips_root=Path(value.get("clips_root", DEFAULT_CLIPS_ROOT)),
        source_videos_root=Path(
            value.get("source_videos_root", DEFAULT_SOURCE_VIDEOS_ROOT)
        ),
        output_root=Path(value.get("output_root", DEFAULT_OUTPUT_ROOT)),
        shard_size=_integer(
            value.get("shard_size", DEFAULT_SHARD_SIZE),
            field_name="shard_size",
            minimum=1,
        ),
        annotation_model=_nonempty_string(
            annotation.get("model"), field_name="annotation.model"
        ),
        api_key=_nonempty_string(
            annotation.get("api_key", "EMPTY"), field_name="annotation.api_key"
        ),
        temperature=_number(
            annotation.get("temperature", 0.0),
            field_name="annotation.temperature",
            minimum=0.0,
        ),
        max_tokens=_integer(
            annotation.get("max_tokens", 4096),
            field_name="annotation.max_tokens",
            minimum=1,
        ),
        timeout_seconds=_integer(
            annotation.get("timeout_seconds", 3600),
            field_name="annotation.timeout_seconds",
            minimum=1,
        ),
        repair_retries=_integer(
            annotation.get("repair_retries", 1),
            field_name="annotation.repair_retries",
            minimum=0,
        ),
        entity_selection_mode=_nonempty_string(
            annotation.get("entity_selection_mode", "default"),
            field_name="annotation.entity_selection_mode",
        ),
        fps=_number(
            annotation.get("fps", 2.0),
            field_name="annotation.fps",
            minimum=0.000001,
        ),
        workers=tuple(workers),
    )
    return settings


def annotation_client_config(
    settings: AnnotationBatchSettings,
    worker: AnnotationBatchWorker,
) -> QwenAnnotationConfig:
    return QwenAnnotationConfig(
        base_url=worker.base_url,
        api_key=settings.api_key,
        model=settings.annotation_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        timeout_seconds=settings.timeout_seconds,
        repair_retries=settings.repair_retries,
        entity_selection_mode=settings.entity_selection_mode,
        video=QwenVideoConfig(fps=settings.fps),
    )


def _json_line(value: object) -> bytes:
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


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _iter_source_rows(path: Path) -> Iterator[SourceRow]:
    resolved = path.expanduser().resolve(strict=True)
    source_index = 0
    with resolved.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                break
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("source JSONL row must contain an object")
            except Exception as exc:  # noqa: BLE001 - preserve one failure row
                yield SourceRow(
                    source_index=source_index,
                    raw=None,
                    read_error=f"{type(exc).__name__}:{exc}",
                )
            else:
                yield SourceRow(source_index=source_index, raw=value)
            source_index += 1


def _output_row_base(
    row: SourceRow,
    *,
    selected: dict[str, object] | None = None,
) -> dict[str, object]:
    raw = row.raw or {}
    return {
        "source_index": row.source_index,
        "clip_uid": selected.get("clip_uid") if selected is not None else None,
        "video_path": (
            selected.get("video_path")
            if selected is not None
            else raw.get("video_path")
        ),
        "source_video_id": raw.get("source_video_id"),
        "source_video_path": raw.get("source_video_path"),
        "shot_index": raw.get("shot_index"),
    }


def _failure_row(
    row: SourceRow,
    reason: str,
    *,
    selected: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        **_output_row_base(row, selected=selected),
        "status": "failed",
        "entities": [],
        "background": None,
        "instruction_template": "",
        "reason": reason[:2048],
        "repair_attempts": 0,
        "warnings": [],
    }


def _ready_row(
    row: SourceRow,
    *,
    selected: dict[str, object],
    annotation: AnnotationState,
    repair_attempts: int,
    warnings: tuple[str, ...],
) -> dict[str, object]:
    if annotation.status != "ready":
        raise ValueError("annotation client returned a non-ready result")
    raw = row.raw or {}
    return {
        "source_index": row.source_index,
        "clip_uid": selected["clip_uid"],
        "video_path": selected["video_path"],
        "source_video_id": raw.get("source_video_id"),
        "source_video_path": raw.get("source_video_path"),
        "shot_index": raw.get("shot_index"),
        "status": annotation.status,
        "entities": [entity.model_dump(mode="json") for entity in annotation.entities],
        "background": (
            annotation.background.model_dump(mode="json")
            if annotation.background is not None
            else None
        ),
        "instruction_template": annotation.instruction_template,
        "reason": annotation.reason,
        "repair_attempts": repair_attempts,
        "warnings": list(warnings),
    }


def _validate_output_row(value: object, *, expected_source_index: int) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("annotation output row must contain an object")
    if value.get("source_index") != expected_source_index:
        raise ValueError("annotation output source_index is not contiguous")
    status = value.get("status")
    entities = value.get("entities")
    if status not in {"ready", "failed"} or not isinstance(entities, list):
        raise ValueError("annotation output status/entities are invalid")
    if status == "ready":
        AnnotationState.model_validate(
            {
                "status": "ready",
                "instruction_template": value.get("instruction_template"),
                "entities": entities,
                "background": value.get("background"),
                "reason": value.get("reason"),
            }
        )
    else:
        AnnotationState.model_validate(
            {
                "status": "failed",
                "reason": value.get("reason"),
            }
        )
        if entities or value.get("background") is not None or value.get(
            "instruction_template"
        ):
            raise ValueError("failed annotation output published semantic content")
    return value


def _read_valid_rows(
    path: Path,
    *,
    expected_indices: list[int],
    recover_incomplete_tail: bool,
) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    mode = "r+b" if recover_incomplete_tail else "rb"
    with path.open(mode) as handle:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if not recover_incomplete_tail:
                    raise ValueError("completed annotation shard has a partial line")
                handle.truncate(line_start)
                handle.flush()
                os.fsync(handle.fileno())
                break
            if len(rows) >= len(expected_indices):
                raise ValueError("annotation shard has more rows than its source range")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("annotation shard contains invalid JSON") from exc
            rows.append(
                _validate_output_row(
                    value,
                    expected_source_index=expected_indices[len(rows)],
                )
            )
    return rows


def _shard_stem(shard_id: int, shard_size: int) -> str:
    start, end = shard_bounds(shard_id, shard_size)
    return f"shard-{start:09d}-{end:09d}"


@contextmanager
def _shard_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise ShardLockedError(f"annotation shard is already locked: {path}") from exc
            raise
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(_json_line(row))
        handle.flush()
        os.fsync(handle.fileno())


def _write_failures(path: Path, rows: list[dict[str, object]]) -> None:
    failures = [row for row in rows if row["status"] == "failed"]
    _atomic_write(path, b"".join(_json_line(row) for row in failures))


def _write_metadata(
    path: Path,
    *,
    settings: AnnotationBatchSettings,
    worker: AnnotationBatchWorker,
    shard_id: int,
    rows: list[dict[str, object]],
) -> None:
    start, nominal_end = shard_bounds(shard_id, settings.shard_size)
    actual_end = int(rows[-1]["source_index"])
    metadata = {
        "shard_id": shard_id,
        "shard_size": settings.shard_size,
        "source_index_start": start,
        "source_index_end": actual_end,
        "nominal_source_index_end": nominal_end,
        "source_jsonl": str(settings.source_jsonl.expanduser().resolve()),
        "base_url": worker.base_url,
        "annotation_model": settings.annotation_model,
        "entity_selection_mode": settings.entity_selection_mode,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write(path, _json_line(metadata))


def _process_source_row(
    row: SourceRow,
    *,
    adapter: JeaVideoMotionAdapter,
    client: AnnotationClient,
) -> dict[str, object]:
    if row.read_error is not None:
        return _failure_row(row, row.read_error)
    assert row.raw is not None
    selected: dict[str, object] | None = None
    try:
        selected = adapter.parse(row.raw, source_index=row.source_index)
        attempt = client.annotate(
            video_path=Path(str(selected["video_path"])),
            caption_raw=str(selected["caption_raw"]),
            metadata=dict(selected["metadata"]),
        )
        return _ready_row(
            row,
            selected=selected,
            annotation=attempt.annotation,
            repair_attempts=attempt.repair_attempts,
            warnings=attempt.warnings,
        )
    except AnnotationFailure as exc:
        reason = (
            exc.issues[0].code if exc.issues else "structured_output_failed"
        )
        return _failure_row(row, reason, selected=selected)
    except Exception as exc:  # noqa: BLE001 - every source row gets an output
        return _failure_row(
            row,
            f"{type(exc).__name__}:{exc}",
            selected=selected,
        )


def _process_shard(
    source_rows: list[SourceRow],
    *,
    shard_id: int,
    settings: AnnotationBatchSettings,
    worker: AnnotationBatchWorker,
    adapter: JeaVideoMotionAdapter,
    client: AnnotationClient,
) -> ShardResult:
    expected_indices = [row.source_index for row in source_rows]
    if not expected_indices:
        raise ValueError("cannot process an empty annotation shard")
    expected_start, expected_end = shard_bounds(shard_id, settings.shard_size)
    if expected_indices[0] != expected_start or expected_indices[-1] > expected_end:
        raise ValueError("source rows do not match annotation shard bounds")
    stem = _shard_stem(shard_id, settings.shard_size)
    part_path = settings.output_root / "parts" / f"{stem}.jsonl"
    partial_path = part_path.with_name(f"{part_path.name}.partial")
    failures_path = settings.output_root / "failures" / f"{stem}.jsonl"
    metadata_path = settings.output_root / "parts" / f"{stem}.meta.json"
    lock_path = settings.output_root / "locks" / f"{stem}.lock"
    with _shard_lock(lock_path):
        if part_path.exists():
            completed = _read_valid_rows(
                part_path,
                expected_indices=expected_indices,
                recover_incomplete_tail=False,
            )
            if len(completed) != len(expected_indices):
                raise ValueError("completed annotation shard is missing source rows")
            return _result(shard_id, part_path, completed, skipped=True)

        completed = _read_valid_rows(
            partial_path,
            expected_indices=expected_indices,
            recover_incomplete_tail=True,
        )
        for row in source_rows[len(completed) :]:
            output = _process_source_row(row, adapter=adapter, client=client)
            _validate_output_row(output, expected_source_index=row.source_index)
            _append_row(partial_path, output)
            completed.append(output)
        if len(completed) != len(expected_indices):
            raise RuntimeError("annotation shard did not produce every source row")
        if part_path.exists():
            raise FileExistsError("completed annotation shard appeared during processing")
        _write_failures(failures_path, completed)
        _write_metadata(
            metadata_path,
            settings=settings,
            worker=worker,
            shard_id=shard_id,
            rows=completed,
        )
        part_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial_path, part_path)
        directory_fd = os.open(part_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return _result(shard_id, part_path, completed, skipped=False)


def _result(
    shard_id: int,
    path: Path,
    rows: list[dict[str, object]],
    *,
    skipped: bool,
) -> ShardResult:
    ready = sum(row["status"] == "ready" for row in rows)
    return ShardResult(
        shard_id=shard_id,
        path=path,
        rows=len(rows),
        ready=ready,
        failed=len(rows) - ready,
        skipped=skipped,
    )


def _append_log(path: Path, value: dict[str, object]) -> None:
    _append_row(path, value)


def run_annotation_batch(
    settings: AnnotationBatchSettings,
    worker: AnnotationBatchWorker,
    *,
    client: AnnotationClient | None = None,
    client_factory: Callable[[QwenAnnotationConfig], AnnotationClient] = (
        QwenAnnotationClient
    ),
) -> dict[str, object]:
    _validate_worker(worker)
    source_jsonl = settings.source_jsonl.expanduser().resolve(strict=True)
    output_root = settings.output_root.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    settings = replace(
        settings,
        source_jsonl=source_jsonl,
        output_root=output_root,
    )
    adapter = JeaVideoMotionAdapter.create(
        clips_root=settings.clips_root,
        source_videos_root=settings.source_videos_root,
    )
    annotation_client = client or client_factory(
        annotation_client_config(settings, worker)
    )
    first_index = worker.shard_start * settings.shard_size
    last_index = (worker.shard_end + 1) * settings.shard_size - 1
    current_shard: int | None = None
    buffered: list[SourceRow] = []
    results: list[ShardResult] = []

    def seal_buffer() -> None:
        nonlocal buffered
        if current_shard is None or not buffered:
            return
        result = _process_shard(
            buffered,
            shard_id=current_shard,
            settings=settings,
            worker=worker,
            adapter=adapter,
            client=annotation_client,
        )
        results.append(result)
        _append_log(
            output_root / "logs" / f"{worker.name}.jsonl",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "worker": worker.name,
                "shard_id": result.shard_id,
                "rows": result.rows,
                "ready": result.ready,
                "failed": result.failed,
                "skipped": result.skipped,
            },
        )
        buffered = []

    for source_row in _iter_source_rows(source_jsonl):
        if source_row.source_index < first_index:
            continue
        if source_row.source_index > last_index:
            seal_buffer()
            break
        shard_id = source_row.source_index // settings.shard_size
        if current_shard is None:
            current_shard = shard_id
        elif shard_id != current_shard:
            seal_buffer()
            current_shard = shard_id
        buffered.append(source_row)
    else:
        seal_buffer()

    return {
        "worker": worker.name,
        "base_url": worker.base_url,
        "shard_start": worker.shard_start,
        "shard_end": worker.shard_end,
        "shards": [
            {
                "shard_id": result.shard_id,
                "path": str(result.path),
                "rows": result.rows,
                "ready": result.ready,
                "failed": result.failed,
                "skipped": result.skipped,
            }
            for result in results
        ],
        "rows": sum(result.rows for result in results),
        "ready": sum(result.ready for result in results),
        "failed": sum(result.failed for result in results),
        "skipped_shards": sum(result.skipped for result in results),
    }


def inspect_annotation_part(path: str | Path) -> dict[str, int | str]:
    resolved = Path(path).expanduser().resolve(strict=True)
    rows = 0
    ready = 0
    failed = 0
    entities = Counter[str]()
    seen_indexes: set[int] = set()
    with resolved.open("rb") as handle:
        for line in handle:
            if not line.endswith(b"\n"):
                raise ValueError("annotation part has an incomplete final line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError("annotation part row must contain an object")
            source_index = value.get("source_index")
            if (
                not isinstance(source_index, int)
                or isinstance(source_index, bool)
                or source_index in seen_indexes
            ):
                raise ValueError("annotation part source indexes are invalid")
            seen_indexes.add(source_index)
            status = value.get("status")
            if status == "ready":
                ready += 1
            elif status == "failed":
                failed += 1
            else:
                raise ValueError("annotation part status is invalid")
            row_entities = value.get("entities")
            if not isinstance(row_entities, list):
                raise ValueError("annotation part entities must contain an array")
            for entity in row_entities:
                if not isinstance(entity, dict):
                    raise ValueError("annotation part entity must contain an object")
                reference_type = entity.get("reference_type")
                if reference_type not in {"subject", "object", "group"}:
                    raise ValueError("annotation part entity reference_type is invalid")
                entities[str(reference_type)] += 1
            rows += 1
    return {
        "path": str(resolved),
        "rows": rows,
        "ready": ready,
        "failed": failed,
        "total_entities": sum(entities.values()),
        "subject": entities["subject"],
        "object": entities["object"],
        "group": entities["group"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--worker")
    parser.add_argument("--base-url")
    parser.add_argument("--shard-start", type=int)
    parser.add_argument("--shard-end", type=int)
    parser.add_argument("--inspect", type=Path)
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if args.inspect is not None:
        if any(
            value is not None
            for value in (
                args.config,
                args.worker,
                args.base_url,
                args.shard_start,
                args.shard_end,
            )
        ):
            raise ValueError("--inspect cannot be combined with batch options")
        result = inspect_annotation_part(args.inspect)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return dict(result)
    if args.config is None or args.worker is None:
        raise ValueError("--config and --worker are required for annotation batches")
    settings = load_batch_settings(args.config)
    worker = select_worker(
        settings,
        args.worker,
        base_url=args.base_url,
        shard_start=args.shard_start,
        shard_end=args.shard_end,
    )
    result = run_annotation_batch(settings, worker)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
