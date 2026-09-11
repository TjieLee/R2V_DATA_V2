"""Read-only static QA for the two target-side TA2VA products."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from urllib.parse import quote, urlsplit

from r2v_data_v2.h3.sam_audio_stem_shadow import _publish_directory, sha256_file
from r2v_data_v2.h3.t2va_shadow import fingerprint, read_rows, render_t2va_prompt
from r2v_data_v2.h3.ta2va_shadow import (
    VERSION,
    TA2VAProduct,
    _verify,
    load_source,
    render_product,
)


def build_ta2va_qa(
    root: Path,
    *,
    media_root: Path | None = None,
    media_base_url: str | None = None,
    overwrite=False,
) -> Path:
    root = root.resolve(strict=True)
    inventory = json.loads((root / "inventory.json").read_text())
    if (
        inventory["schema_version"] != VERSION
        or inventory["output_root"] != str(root)
        or inventory["inventory_fingerprint"]
        != fingerprint(
            {k: v for k, v in inventory.items() if k != "inventory_fingerprint"}
        )
    ):
        raise ValueError("TA2VA QA inventory ownership differs")
    artifacts = json.loads((root / "artifacts.json").read_text())
    for relative, digest in artifacts.items():
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError("TA2VA published artifact hash differs")
    _verify(inventory["source_hashes"])
    source, clips, _, hashes = load_source(Path(inventory["source_t2va_root"]))
    if (
        source.inventory_fingerprint != inventory["source_inventory_fingerprint"]
        or hashes != inventory["source_hashes"]
    ):
        raise ValueError("TA2VA QA source lineage differs")
    products = read_rows(root / "records.jsonl", TA2VAProduct)
    if len({(p.clip_uid, p.variant) for p in products}) != len(products):
        raise ValueError("duplicate TA2VA product")
    if (media_root is None) != (media_base_url is None):
        raise ValueError("QA media root/base URL must be supplied together")
    if media_base_url:
        parsed = urlsplit(media_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("QA media URL must be HTTP(S)")
        media_root = media_root.resolve(strict=True)
    destination = root / "qa"
    if destination.resolve() != destination:
        raise ValueError("TA2VA QA output redirects")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)

    def url(path):
        path = Path(path).resolve(strict=True)
        if media_base_url:
            return (
                media_base_url.rstrip("/")
                + "/"
                + quote(path.relative_to(media_root).as_posix())
            )
        return quote(os.path.relpath(path, destination), safe="/")

    cases = []
    for job, core, _, core_hash in clips:
        variants = []
        for p in products:
            if p.clip_uid != job.clip_uid:
                continue
            if (
                p.source_core_sha256 != core_hash
                or p.source_inventory_fingerprint != source.inventory_fingerprint
                or p.source_resolved_record_fingerprint
                != job.audio_evidence.resolved_record_fingerprint
            ):
                raise ValueError("TA2VA product core differs")
            if p.status == "ready" and p.prompt != render_product(
                core, p.variant, p.audio_references
            ):
                raise ValueError("TA2VA prompt differs from frozen core projection")
            refs = []
            for ref in p.audio_references:
                _verify({ref.path: ref.sha256})
                asset = next((a for a in p.assets if a.output_path == ref.path), None)
                if ref.kind == "full_audio_reuse":
                    if (ref.path, ref.sha256) != (
                        job.audio_evidence.full_audio_path,
                        job.audio_evidence.full_audio_sha256,
                    ):
                        raise ValueError(
                            "TA2VA full Audio differs from canonical target"
                        )
                elif asset is None or (asset.output_sha256, asset.kind) != (
                    ref.sha256,
                    ref.kind,
                ):
                    raise ValueError("TA2VA reuse asset differs from Audio contract")
                if asset:
                    _verify({asset.source_path: asset.source_sha256})
                refs.append(
                    {
                        "contract": ref.model_dump(mode="json"),
                        "url": url(ref.path),
                        "asset": asset.model_dump(mode="json") if asset else None,
                    }
                )
            variants.append({**p.model_dump(mode="json"), "references": refs})
        raw_path = root / "raw_profiles" / f"{job.clip_uid}.json"
        cases.append(
            {
                "clip_uid": job.clip_uid,
                "video_url": url(job.target_video_path),
                "full_audio_url": url(job.audio_evidence.full_audio_path),
                "t2va_prompt": render_t2va_prompt(core),
                "speech": [
                    {
                        **f.model_dump(mode="json"),
                        "assignment": a.model_dump(mode="json"),
                    }
                    for f, a in zip(
                        core.speech_facts, core.speaker_assignments, strict=True
                    )
                ],
                "variants": variants,
                "profile_raw": json.loads(raw_path.read_text())
                if raw_path.exists()
                else None,
            }
        )
    payload = {
        "schema_version": VERSION,
        "run_root": str(root),
        "cases": cases,
        "summary": json.loads((root / "summary.json").read_text()),
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    escaped = (
        data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    html = Path(__file__).with_suffix(".html").read_text().replace("__DATA__", escaped)
    temporary = root / f".qa.tmp-{uuid.uuid4().hex}"
    try:
        temporary.mkdir()
        (temporary / "data.json").write_text(data + "\n")
        (temporary / "review.html").write_text(html)
        _verify(inventory["source_hashes"])
        _publish_directory(temporary, destination, overwrite=overwrite)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination / "review.html"
