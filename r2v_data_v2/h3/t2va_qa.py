"""Static, read-only QA for the independent T2VA namespace."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit

from r2v_data_v2.h3.sam_audio_stem_shadow import _publish_directory, sha256_file
from r2v_data_v2.h3.t2va_shadow import H3NoReferenceAVCore, load_t2va_shadow, t2va_root


def build_t2va_qa(
    run_root: Path,
    *,
    media_root: Path | None = None,
    media_base_url: str | None = None,
    overwrite: bool = False,
) -> Path:
    root = run_root.expanduser().resolve(strict=True)
    inventory, records, summary = load_t2va_shadow(root)
    if root != t2va_root(Path(inventory.audio_production_root), inventory.t2va_run_id):
        raise ValueError("T2VA QA requires the owned published run root")
    if (media_root is None) != (media_base_url is None):
        raise ValueError("QA media root and base URL must be supplied together")
    if media_base_url is not None:
        parsed = urlsplit(media_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("QA media base URL must be HTTP(S)")
        media_root = media_root.expanduser().resolve(strict=True)
    destination = root / "qa"
    if destination.resolve() != destination:
        raise ValueError("T2VA QA output must not redirect")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(destination)
        old = json.loads((destination / "data.json").read_text())
        if old.get("schema_version") != "r2v.h3.t2va_qa.1" or old.get(
            "run_root"
        ) != str(root):
            raise ValueError("T2VA QA overwrite ownership differs")
    cases = []
    for job, record in zip(inventory.jobs, records, strict=True):
        video = Path(job.target_video_path).resolve(strict=True)
        if sha256_file(video) != job.target_video_sha256:
            raise ValueError("T2VA QA original target video changed")
        if media_base_url:
            url = (
                media_base_url.rstrip("/")
                + "/"
                + quote(video.relative_to(media_root).as_posix())
            )
        else:
            url = quote(os.path.relpath(video, destination), safe="/")
        assignments = {}
        prompt = None
        if record.status == "ready":
            core = H3NoReferenceAVCore.model_validate_json(
                (root / "core" / f"{job.clip_uid}.json").read_text()
            )
            assignments = {
                a.segment_id: a.model_dump(mode="json")
                for a in core.speaker_assignments
            }
            prompt = (root / "prompts" / f"{job.clip_uid}.txt").read_text()
        cases.append(
            {
                "clip_uid": job.clip_uid,
                "clip_display_path": job.clip_display_path,
                "video_url": url,
                "status": record.status,
                "failure_reason": record.failure_reason,
                "prompt": prompt,
                "warnings": record.warnings,
                "speech": [
                    {
                        **s.model_dump(mode="json"),
                        "assignment": assignments.get(s.segment_id),
                    }
                    for s in job.speech_facts
                ],
                "raw": json.loads((root / "raw" / f"{job.clip_uid}.json").read_text()),
            }
        )
    payload = {
        "schema_version": "r2v.h3.t2va_qa.1",
        "run_root": str(root),
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "summary": summary.model_dump(mode="json"),
        "cases": cases,
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    embedded = (
        data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    html = (
        Path(__file__)
        .with_suffix(".html")
        .read_text()
        .replace("__T2VA_DATA__", embedded)
    )
    temporary = destination.with_name(f".qa.tmp-{uuid.uuid4().hex}")
    try:
        temporary.mkdir()
        (temporary / "data.json").write_text(data + "\n", encoding="utf-8")
        (temporary / "review.html").write_text(html, encoding="utf-8")
        _publish_directory(temporary, destination, overwrite=overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "review.html"
