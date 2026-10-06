"""Read-only viewer for effective single-v7 RA2VA reconcile records."""

from __future__ import annotations

import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


def _records(root: Path) -> dict[str, dict]:
    records = {}
    with (root / "records.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["clip_uid"]] = record
    return records


def build_caption_cases(base_root: Path, override_root: Path | None = None) -> list[dict]:
    """Overlay records by UID while keeping BASE source jobs and their order."""
    source = json.loads((base_root / "source_contract.json").read_text(encoding="utf-8"))
    records = _records(base_root)
    if override_root is not None:
        records.update(_records(override_root))

    jobs = {job["clip_uid"]: job for job in source["jobs"]}
    cases = []
    for uid in source["clip_uids"]:
        job = jobs[uid]
        record = records.get(uid, {})
        raw = (record.get("raw_responses") or [None])[0]
        annotation = record.get("annotation") or {}
        caption = annotation.get("h3_semantics") or {}
        if not caption and raw:
            try:
                caption = json.loads(raw).get("h3_semantics") or {}
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        subjects = job["reference_subjects"]
        references = [
            {
                "picture_label": image["picture_label"],
                "subject_labels": [
                    subject["subject_label"]
                    for subject in subjects
                    if image["picture_label"] in subject["source_picture_labels"]
                ],
                "kind": image["kind"],
                "path": image["image_artifact_path"],
            }
            for image in job["reference_images"]
        ]
        cases.append(
            {
                "clip_uid": uid,
                "status": record.get("status", "missing"),
                "failure_reason": record.get("failure_reason"),
                "video_path": job["target_video_path"],
                "audio_path": job.get("target_full_audio_path"),
                "references": references,
                "subjects": subjects,
                "segments": job["segments"],
                "groundings": (annotation.get("av_grounding") or {}).get("segment_groundings", []),
                "caption": caption,
                "raw_compact": raw,
            }
        )
    return cases


def _media_payload(cases: list[dict]) -> tuple[list[dict], dict[str, Path]]:
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
        item = dict(case)
        item["video_url"] = url(case["video_path"])
        item["audio_url"] = url(case["audio_path"]) if case["audio_path"] else None
        item["references"] = [
            {**reference, "url": url(reference["path"])}
            for reference in case["references"]
        ]
        payload.append(item)
    return payload, media


def _html(payload: list[dict]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    escaped = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    template = Path(__file__).with_suffix(".html").read_text(encoding="utf-8")
    return template.replace("__DATA__", escaped).encode("utf-8")


_MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
}


def make_caption_server(*, host: str, port: int, cases: list[dict]) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError("single-v7 caption viewer requires a loopback host")
    payload, media = _media_payload(cases)
    page = _html(payload)

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
                    self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "text/plain")
                    return
                start = int(match.group(1))
                end = min(size - 1, int(match.group(2)) if match.group(2) else size - 1)
                status = HTTPStatus.PARTIAL_CONTENT
            if start > end:
                self._send(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "text/plain")
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
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
                return
            match = re.fullmatch(r"/media/(m\d+)", path)
            media_path = media.get(match.group(1)) if match else None
            if media_path is None:
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            self._serve_media(media_path)

        def do_POST(self) -> None:
            self._send(HTTPStatus.METHOD_NOT_ALLOWED, b"read only", "text/plain")

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
