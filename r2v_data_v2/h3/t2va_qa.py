"""Static, read-only QA for the independent T2VA namespace."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit

from r2v_data_v2.h3.sam_audio_stem_shadow import _publish_directory, sha256_file
from r2v_data_v2.h3.t2va_shadow import (
    H3NoReferenceAVCore,
    check_audio_files,
    load_t2va_shadow,
    render_t2va_speech,
    t2va_root,
    validate_t2va_audio_lineage,
)

QA_VERSION = "r2v.h3.t2va_qa.3"


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
    if any(job.audio_evidence is not None for job in inventory.jobs):
        validate_t2va_audio_lineage(inventory)
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
        if old.get("schema_version") not in {
            "r2v.h3.t2va_qa.1",
            "r2v.h3.t2va_qa.2",
            QA_VERSION,
        } or old.get("run_root") != str(root):
            raise ValueError("T2VA QA overwrite ownership differs")

    def media_url(path: Path) -> str:
        path = path.resolve(strict=True)
        if media_base_url:
            return (
                media_base_url.rstrip("/")
                + "/"
                + quote(path.relative_to(media_root).as_posix())
            )
        return quote(os.path.relpath(path, destination), safe="/")

    cases = []
    for job, record in zip(inventory.jobs, records, strict=True):
        video = Path(job.target_video_path).resolve(strict=True)
        if sha256_file(video) != job.target_video_sha256:
            raise ValueError("T2VA QA original target video changed")
        media = {}
        if job.audio_evidence is not None:
            check_audio_files(job.audio_evidence)
            for kind in ("full_audio", "speech", "music", "sfx"):
                media[kind] = {
                    "url": media_url(Path(getattr(job.audio_evidence, f"{kind}_path"))),
                    "sha256": getattr(job.audio_evidence, f"{kind}_sha256"),
                }
        assignments = {}
        placements = {}
        rendered = {}
        prompt = None
        finalized_audio = None
        if record.status == "ready":
            core = H3NoReferenceAVCore.model_validate_json(
                (root / "core" / f"{job.clip_uid}.json").read_text()
            )
            assignments = {
                a.segment_id: a.model_dump(mode="json")
                for a in core.speaker_assignments
            }
            placements = {
                p.segment_id: p for p in core.integrated_sequence if p.kind == "speech"
            }
            for fact, assignment in zip(
                core.speech_facts, core.speaker_assignments, strict=True
            ):
                if fact.segment_id in placements:
                    rendered[fact.segment_id] = render_t2va_speech(
                        fact, assignment, placements[fact.segment_id]
                    )
            prompt = (root / "prompts" / f"{job.clip_uid}.txt").read_text()
            finalized_audio = {
                "overall_soundscape": core.overall_soundscape,
                "non_diegetic_music": core.non_diegetic_music,
                "prompt_version": inventory.backend.audio_finalize_prompt_version,
            }
        raw = json.loads((root / "raw" / f"{job.clip_uid}.json").read_text())
        cases.append(
            {
                "clip_uid": job.clip_uid,
                "clip_display_path": job.clip_display_path,
                "video_url": media_url(video),
                "audio_media": media,
                "audio_lineage": job.audio_evidence.model_dump(mode="json")
                if job.audio_evidence
                else None,
                "finalized_audio": finalized_audio,
                "status": record.status,
                "failure_reason": record.failure_reason,
                "prompt": prompt,
                "warnings": record.warnings,
                "speech": [
                    {
                        **s.model_dump(mode="json"),
                        "assignment": assignments.get(s.segment_id),
                        "model_lead_in": placements[s.segment_id].lead_in
                        if s.segment_id in placements
                        else None,
                        "rendered_speech": rendered.get(s.segment_id),
                    }
                    for s in job.speech_facts
                ],
                "raw": raw,
                "semantic_raw": {
                    key: raw[key]
                    for key in (
                        "response",
                        "finish_reason",
                        "usage",
                        "warnings",
                        "semantic_model_call_count",
                        "deterministic_corrections",
                    )
                },
                "audio_finalize_raw": raw["audio_finalize"],
                "pipeline_error": raw["error"],
            }
        )
    payload = {
        "schema_version": QA_VERSION,
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
