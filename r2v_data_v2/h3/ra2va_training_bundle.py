"""Deterministic frozen-reconcile to reviewable R(A)2VA task bundle."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from r2v_data_v2.h3.audio_reuse import build_audio_reuse_assets
from r2v_data_v2.h3.audio_reuse_finalizer import (
    ASSETS_STAGE,
    PREPARED_STAGE,
    PRODUCTS_STAGE,
)
from r2v_data_v2.h3.audio_reuse_materializer import (
    AudioReuseProduct,
    materialize_audio_reuse_products,
)
from r2v_data_v2.h3.audio_reuse_prepared import (
    AudioReusePreparedSource,
    load_prepared_inventory,
    prepare_frozen_audio_reuse_sources,
)
from r2v_data_v2.h3.frame_conditioning_projection import (
    materialize_frame_conditioned_products,
)
from r2v_data_v2.h3.resolved_audio_stems import load_stem_source
from r2v_data_v2.h3.training_manifest_export import build_ra2va_training_task_rows


def materialize_ra2va_training_bundle(
    *, reconcile_root: Path, output_root: Path, clip_uids: list[str] | None = None,
    source_h3_root: Path | None = None, allow_unverified: bool = False, ffmpeg: str = "ffmpeg",
) -> dict:
    """Publish new owned stages only; human review remains a separate operation."""
    output = output_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    prepared = output / PREPARED_STAGE
    prepare_frozen_audio_reuse_sources(
        reconcile_root=reconcile_root, prepared_root=prepared, clip_uids=clip_uids, source_h3_root=source_h3_root,
    )
    inventory = load_prepared_inventory(prepared / "inventory.json")
    stem_root = Path(inventory.source_stem_root)
    stem_inventory, stems, _ = load_stem_source(stem_root)
    audio_root = Path(stem_inventory.source_canonical_audio_manifest_path).parent.parent
    by_clip = {r.clip_uid: r for r in stems}
    records = [AudioReusePreparedSource.model_validate_json(line)
               for line in (prepared / "records.jsonl").read_text().splitlines() if line.strip()]
    for job, record in zip(inventory.jobs, records, strict=True):
        build_audio_reuse_assets(
            job=job, annotation=record.annotation, stem_record=by_clip[job.clip_uid],
            audio_production_root=audio_root, output_root=output / ASSETS_STAGE / job.clip_uid,
            allow_unverified=allow_unverified,
        )
    products = materialize_audio_reuse_products(
        mimo_root=prepared, source_h3_root=prepared / "h3", separation_root=stem_root,
        reuse_root=output / ASSETS_STAGE, audio_production_root=audio_root, output_root=output / PRODUCTS_STAGE,
        enable_full_audio_reuse=True,
    )
    frames = materialize_frame_conditioned_products(source_root=output, ffmpeg=ffmpeg)
    product_records = [AudioReuseProduct.model_validate_json(line)
                       for line in (output / PRODUCTS_STAGE / "records.jsonl").read_text().splitlines() if line.strip()]
    variants = Counter(r.conditioning_variant for r in product_records if r.status == "ready")
    tasks = build_ra2va_training_task_rows(output)
    summary = {
        "source_reconcile_root": str(reconcile_root.expanduser().resolve()),
        "clip_uids": [j.clip_uid for j in inventory.jobs],
        "product_ready_count": products.ready_count, "product_failed_count": products.failed_count,
        "conditioning_variant_counts": dict(variants),
        "unavailable_target_speech_reuse_clip_uids": [
            j.clip_uid for j in inventory.jobs if not any(
                r.clip_uid == j.clip_uid and r.conditioning_variant == "target_speech_reuse" for r in product_records
            )
        ],
        "frame_product_count": frames.derived_product_count,
        "task_counts": {task: len(rows) for task, rows in tasks.items()},
        "model_call_count": 0, "production_artifacts_modified": False,
        "human_review_required": True,
    }
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    temporary.rename(output / "summary.json")
    return summary
