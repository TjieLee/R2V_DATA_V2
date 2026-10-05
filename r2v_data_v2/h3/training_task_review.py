"""Read-only R(A)2VA training review with a separate human annotation sidecar."""

from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import threading
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.training_manifest_export import (
    RA2VA_TASK_ORDER,
    build_ra2va_training_task_rows,
    write_training_manifests,
)

REVIEW_TASKS = RA2VA_TASK_ORDER
ANNOTATION_VERSION = "r2v.h3.training_task_review.1"
SUMMARY_VERSION = "r2v.h3.training_task_review_summary.1"
ISSUE_TAGS = (
    "caption_visual_issue",
    "subject_reference_issue",
    "picture_usage_issue",
    "speaker_binding_issue",
    "speaker_grouping_issue",
    "dialogue_issue",
    "audio_reference_issue",
    "voice_profile_issue",
    "soundscape_issue",
    "music_issue",
    "conditioning_mismatch",
    "other",
)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


class TrainingTaskReviewAnnotation(SchemaModel):
    schema_version: Literal["r2v.h3.training_task_review.1"] = ANNOTATION_VERSION
    clip_uid: str = Field(min_length=1)
    task: str
    sample_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["PASS", "ISSUE", "SKIP"]
    issue_tags: list[str]
    notes: str
    reviewed_at: str

    @model_validator(mode="after")
    def validate_annotation(self) -> TrainingTaskReviewAnnotation:
        if self.task not in REVIEW_TASKS:
            raise ValueError("unknown R(A)2VA training task")
        if len(self.issue_tags) != len(set(self.issue_tags)) or any(
            tag not in ISSUE_TAGS for tag in self.issue_tags
        ):
            raise ValueError("invalid training review issue tags")
        if self.decision == "ISSUE" and not self.issue_tags:
            raise ValueError("ISSUE review requires at least one issue tag")
        if self.decision != "ISSUE" and self.issue_tags:
            raise ValueError("only ISSUE review may publish issue tags")
        return self


class TrainingTaskReviewSummary(SchemaModel):
    schema_version: Literal["r2v.h3.training_task_review_summary.1"] = SUMMARY_VERSION
    current_task_count: int = Field(ge=0)
    reviewed_count: int = Field(ge=0)
    unreviewed_count: int = Field(ge=0)
    decision_counts: dict[str, int]
    issue_tag_counts: dict[str, int]
    stale_annotation_count: int = Field(ge=0)


@dataclass(frozen=True)
class TrainingTaskReviewCase:
    clip_uid: str
    task: str
    row: dict
    image_labels: tuple[str, ...]
    audio_kinds: tuple[str, ...]
    sample_fingerprint: str


def build_review_cases(shadow_root: Path) -> list[TrainingTaskReviewCase]:
    rows = build_ra2va_training_task_rows(shadow_root.expanduser())
    cases = []
    seen = set()
    for task in REVIEW_TASKS:
        for item in rows[task]:
            if not isinstance(item.clip_uid, str) or not item.clip_uid:
                raise ValueError("R(A)2VA review task requires clip_uid")
            key = (item.clip_uid, task)
            if key in seen:
                raise ValueError(f"duplicate R(A)2VA review task: {key}")
            seen.add(key)
            fingerprint = hashlib.sha256(
                _compact_json(
                    {"clip_uid": item.clip_uid, "task": task, "row": item.row}
                ).encode("utf-8")
            ).hexdigest()
            cases.append(
                TrainingTaskReviewCase(
                    clip_uid=item.clip_uid,
                    task=task,
                    row=item.row,
                    image_labels=item.image_labels,
                    audio_kinds=item.audio_kinds,
                    sample_fingerprint=fingerprint,
                )
            )
    task_index = {task: index for index, task in enumerate(REVIEW_TASKS)}
    return sorted(cases, key=lambda item: (item.clip_uid, task_index[item.task]))


class TrainingTaskReviewStore:
    def __init__(self, root: Path, cases: Sequence[TrainingTaskReviewCase]) -> None:
        self.root = root.expanduser().resolve(strict=False)
        self.cases = list(cases)
        self.current = {
            (item.clip_uid, item.task): item.sample_fingerprint for item in cases
        }
        if len(self.current) != len(self.cases):
            raise ValueError("duplicate R(A)2VA review task")
        self._lock = threading.RLock()

    @property
    def annotations_path(self) -> Path:
        return self.root / "annotations.jsonl"

    def load_all(self) -> list[TrainingTaskReviewAnnotation]:
        if not self.annotations_path.is_file():
            return []
        rows = [
            TrainingTaskReviewAnnotation.model_validate_json(line)
            for line in self.annotations_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len({(item.clip_uid, item.task) for item in rows}) != len(rows):
            raise ValueError("duplicate training review annotation")
        return rows

    def current_annotations(
        self,
    ) -> dict[tuple[str, str], TrainingTaskReviewAnnotation]:
        return {
            (item.clip_uid, item.task): item
            for item in self.load_all()
            if self.current.get((item.clip_uid, item.task)) == item.sample_fingerprint
        }

    def stale_keys(self) -> set[tuple[str, str]]:
        return {
            (item.clip_uid, item.task)
            for item in self.load_all()
            if (item.clip_uid, item.task) in self.current
            and self.current[(item.clip_uid, item.task)] != item.sample_fingerprint
        }

    def save(
        self, annotation: TrainingTaskReviewAnnotation
    ) -> TrainingTaskReviewSummary:
        key = (annotation.clip_uid, annotation.task)
        if self.current.get(key) != annotation.sample_fingerprint:
            raise ValueError("training review annotation is stale or unknown")
        with self._lock:
            by_task = {(item.clip_uid, item.task): item for item in self.load_all()}
            by_task[key] = annotation
            _atomic_text(
                self.annotations_path,
                "".join(
                    _compact_json(by_task[item].model_dump(mode="json")) + "\n"
                    for item in sorted(by_task)
                ),
            )
            return self.publish_derived()

    def publish_derived(self) -> TrainingTaskReviewSummary:
        with self._lock:
            all_rows = self.load_all()
            current = [
                item
                for item in all_rows
                if self.current.get((item.clip_uid, item.task))
                == item.sample_fingerprint
            ]
            decisions = Counter(item.decision for item in current)
            issues = Counter(tag for item in current for tag in item.issue_tags)
            summary = TrainingTaskReviewSummary(
                current_task_count=len(self.cases),
                reviewed_count=len(current),
                unreviewed_count=len(self.cases) - len(current),
                decision_counts=dict(sorted(decisions.items())),
                issue_tag_counts=dict(sorted(issues.items())),
                stale_annotation_count=len(all_rows) - len(current),
            )
            _atomic_text(
                self.root / "summary.json",
                json.dumps(
                    summary.model_dump(mode="json"), ensure_ascii=False, indent=2
                )
                + "\n",
            )
            csv_path = self.root / "annotations.csv"
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{csv_path.name}.", dir=self.root
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(
                        (
                            "clip_uid",
                            "task",
                            "sample_fingerprint",
                            "decision",
                            "issue_tags",
                            "notes",
                            "reviewed_at",
                        )
                    )
                    for item in sorted(
                        current, key=lambda row: (row.clip_uid, row.task)
                    ):
                        writer.writerow(
                            (
                                item.clip_uid,
                                item.task,
                                item.sample_fingerprint,
                                item.decision,
                                "|".join(item.issue_tags),
                                item.notes,
                                item.reviewed_at,
                            )
                        )
                Path(temporary).replace(csv_path)
            except Exception:
                Path(temporary).unlink(missing_ok=True)
                raise
            return summary


def export_reviewed_training_manifests(
    *,
    cases: Sequence[TrainingTaskReviewCase],
    store: TrainingTaskReviewStore,
    output_root: Path,
) -> Path:
    if {
        (case.clip_uid, case.task): case.sample_fingerprint for case in cases
    } != store.current:
        raise ValueError("review export cases differ from current review store")
    annotations = store.current_annotations()
    rows = {task: [] for task in REVIEW_TASKS}
    for case in cases:
        annotation = annotations.get((case.clip_uid, case.task))
        if annotation is not None and annotation.decision == "PASS":
            rows[case.task].append(case.row)
    return write_training_manifests(
        output_root=output_root, rows=rows, task_order=REVIEW_TASKS
    )


def _media_payload(
    cases: Sequence[TrainingTaskReviewCase],
) -> tuple[list[dict], dict[str, Path]]:
    media: dict[str, Path] = {}
    by_path: dict[str, str] = {}

    def url(path: str) -> str:
        if path not in by_path:
            token = f"m{len(media)}"
            by_path[path] = token
            media[token] = Path(path).expanduser()
        return "/media/" + by_path[path]

    payload = []
    for case in cases:
        payload.append(
            {
                "clip_uid": case.clip_uid,
                "task": case.task,
                "row": case.row,
                "image_labels": case.image_labels,
                "audio_kinds": case.audio_kinds,
                "sample_fingerprint": case.sample_fingerprint,
                "video_url": url(case.row["video"]),
                "image_urls": [url(path) for path in case.row["images"]],
                "audio_urls": [url(path) for path in case.row["audios"]],
            }
        )
    return payload, media


def _html(payload: list[dict], store: TrainingTaskReviewStore) -> str:
    annotations = {
        f"{clip_uid}|{task}": item.model_dump(mode="json")
        for (clip_uid, task), item in store.current_annotations().items()
    }
    stale = [f"{clip_uid}|{task}" for clip_uid, task in sorted(store.stale_keys())]
    data = json.dumps(
        {
            "cases": payload,
            "annotations": annotations,
            "stale": stale,
            "issue_tags": ISSUE_TAGS,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    escaped = (
        data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    return template.replace("__DATA__", escaped)


_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
}


def make_review_server(
    *,
    host: str,
    port: int,
    cases: Sequence[TrainingTaskReviewCase],
    store: TrainingTaskReviewStore,
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("training review server requires a loopback host")
    payload, media = _media_payload(cases)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_media(self, path: Path) -> None:
            if not path.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            status = HTTPStatus.OK
            range_header = self.headers.get("Range")
            if range_header is not None:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
                if match is None:
                    self._send(
                        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "text/plain"
                    )
                    return
                start = int(match.group(1))
                end = min(size - 1, int(match.group(2)) if match.group(2) else size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
            if start > end:
                self._send(
                    HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "text/plain"
                )
                return
            length = end - start + 1
            content_type = (
                _MEDIA_TYPES.get(path.suffix.lower())
                or mimetypes.guess_type(path.name)[0]
                or "application/octet-stream"
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = length
                while remaining:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/":
                self._send(
                    HTTPStatus.OK,
                    _html(payload, store).encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            match = re.fullmatch(r"/media/(m\d+)", path)
            media_path = None if match is None else media.get(match.group(1))
            if media_path is None:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            self._serve_media(media_path)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/annotation":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64 * 1024:
                    raise ValueError("invalid annotation payload length")
                annotation = TrainingTaskReviewAnnotation.model_validate_json(
                    self.rfile.read(length)
                )
                summary = store.save(annotation)
                self._send(
                    HTTPStatus.OK,
                    _compact_json(summary.model_dump(mode="json")).encode("utf-8"),
                    "application/json",
                )
            except (OSError, ValueError) as exc:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    str(exc).encode("utf-8"),
                    "text/plain; charset=utf-8",
                )

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
