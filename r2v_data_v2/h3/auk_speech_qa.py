"""Static, independent comparison of frozen AuK and SAM candidates."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from r2v_data_v2.h3.auk_speech_shadow import (
    FIXED_INSTRUCTION,
    auk_stage_root,
    check_source,
    fingerprint,
    load_auk_shadow,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    load_stem_shadow,
    sha256_file,
    stem_separation_root,
    stem_shadow_root,
)


def build_auk_speech_qa(*, audio_production_root: Path, shadow_run_id: str) -> dict:
    root = stem_shadow_root(audio_production_root, shadow_run_id)
    destination = root / "auk_speech_qa_v1"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    inventory, records, _ = load_auk_shadow(
        auk_stage_root(audio_production_root, shadow_run_id)
    )
    sam, sam_records, _ = load_stem_shadow(
        stem_separation_root(audio_production_root, shadow_run_id)
    )
    config = sam.model_configuration
    if (
        sam.route,
        sam.run_both_routes,
        config.speech_prompt,
        config.music_prompt,
        config.predict_spans,
    ) != (
        "voice_first",
        False,
        "human voices",
        "music soundtrack",
        False,
    ):
        raise ValueError(
            "AuK QA requires the voice_first / human voices / no spans SAM baseline"
        )
    baseline = {r.clip_uid: r for r in sam_records}
    if (
        len(baseline) != len(sam_records)
        or [r.clip_uid for r in sam_records] != sam.clip_uids
    ):
        raise ValueError("SAM baseline record inventory differs")
    sam_jobs = {j.clip_uid: j for j in sam.jobs}
    links: dict[str, Path] = {}

    def media(path: str, expected: str) -> str:
        source = Path(path).resolve(strict=True)
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError("AuK QA media hash differs")
        url = f"media/{fingerprint([str(source), expected])}{source.suffix.lower()}"
        links[url] = source
        return url

    cases = []
    for job, record in zip(inventory.jobs, records, strict=True):
        check_source(job)
        baseline_job = sam_jobs.get(job.clip_uid)
        baseline_record = baseline.get(job.clip_uid)
        if (
            baseline_job is None
            or baseline_record is None
            or baseline_job.model_dump() != job.model_dump()
        ):
            raise ValueError("AuK QA source lineage differs from SAM baseline")
        if (
            baseline_record.route != "voice_first"
            or baseline_record.model_configuration_fingerprint
            != config.configuration_fingerprint
        ):
            raise ValueError("SAM baseline configuration lineage differs")
        row = {
            "clip_uid": job.clip_uid,
            "clip_display_path": job.clip_display_path,
            "original": media(job.target_video_path, job.target_video_sha256),
            "source_audio_sha256": job.source_audio_sha256,
            "auk_record_fingerprint": record.record_fingerprint,
            "sam_record_fingerprint": baseline_record.record_fingerprint,
            "auk_status": record.status,
            "auk_failure": record.failure_reason,
            "sam_status": baseline_record.separation_state,
            "sam_failure": baseline_record.failure_reason,
            "auk": None,
            "sam_speech": None,
            "sam_sfx": None,
        }
        if record.canonical is not None:
            row["auk"] = media(record.canonical.path, record.canonical.sha256)
        if baseline_record.separation_state != "failure":
            for kind in ("speech", "sfx"):
                stem = baseline_record.stem(kind)
                if (stem.source_audio_sha256, stem.source_end_sample) != (
                    job.source_audio_sha256,
                    job.source_frame_count,
                ):
                    raise ValueError("SAM stem source lineage differs")
                row[f"sam_{kind}"] = media(
                    stem.canonical_stem_path, stem.canonical_stem_sha256
                )
        cases.append(row)
    payload = {
        "schema_version": "r2v.h3.auk_speech_qa.1",
        "cases": cases,
        "instruction": FIXED_INSTRUCTION,
        "auk_configuration": inventory.model_configuration.model_dump(mode="json"),
        "auk_inventory_fingerprint": inventory.inventory_fingerprint,
        "sam_inventory_fingerprint": sam.inventory_fingerprint,
        "sam_configuration_fingerprint": config.configuration_fingerprint,
        "production_artifacts_modified": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".auk-qa-", dir=root))
    try:
        (temporary / "media").mkdir()
        for url, source in links.items():
            (temporary / url).symlink_to(source)
        data = json.dumps(payload, sort_keys=True, ensure_ascii=False).replace(
            "<", "\\u003c"
        )
        template = Path(__file__).with_name("auk_speech_qa.html").read_text()
        (temporary / "index.html").write_text(
            template.replace("__AUK_QA_DATA__", data), encoding="utf-8"
        )
        (temporary / "data.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {"output_root": str(destination), "clip_count": len(cases)}
