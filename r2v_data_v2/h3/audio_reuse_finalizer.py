"""Model-free, named-run orchestration of the existing Audio reuse stages."""
from __future__ import annotations

import tempfile
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.audio_reuse import build_audio_reuse_assets
from r2v_data_v2.h3.audio_reuse_materializer import materialize_audio_reuse_products
from r2v_data_v2.h3.audio_reuse_prepared import (
    load_reconcile_sources,
    prepare_audio_reuse_sources,
    project_prepared_samples,
    validate_prepared_inputs,
)
from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoCaseManifest,
    _inventory,
    build_mimo25_inventory,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MIMO25_STEM_RECONCILE_STAGE,
    MimoStemReconcileSummary,
    build_stem_reconcile_jobs,
    usable_stem_reconcile_inventory,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    SAMRoute,
    StemRoute,
    require_shadow_output_path,
    selected_stem_records,
    sha256_file,
    stem_shadow_root,
    validate_stem_diarization_lineage,
)
from r2v_data_v2.h3.schemas import SchemaModel

ASSETS_STAGE = "audio_reuse_assets_v1"
PREPARED_STAGE = "audio_reuse_prepared_v1"
PRODUCTS_STAGE = "h3_audio_reuse_products_v1"
FINALIZATION_STAGE = "audio_reuse_finalization_v1"


class AudioReuseFinalizationSummary(SchemaModel):
    schema_version: Literal["r2v.h3.audio_reuse_finalization_summary.1", "r2v.h3.audio_reuse_finalization_summary.2"] = "r2v.h3.audio_reuse_finalization_summary.1"
    shadow_run_id: str
    sam_route: StemRoute
    allow_unverified: bool
    clip_uids: list[str]
    clip_count: int = Field(ge=0)
    reconcile_ready_count: int = Field(ge=0)
    reconcile_failed_count: int = Field(ge=0)
    upstream_unavailable_count: int = Field(ge=0)
    unavailable_clips: dict[str, str]
    reuse_asset_clip_count: int = Field(ge=0)
    speaker_speech_asset_count: int = Field(ge=0)
    music_asset_count: int = Field(ge=0)
    reuse_exclusion_reason_counts: dict[str, int]
    prepared_ready_count: int = Field(ge=0)
    prepared_failed_count: int = Field(ge=0)
    product_sample_count: int = Field(ge=0)
    product_ready_count: int = Field(ge=0)
    product_failed_count: int = Field(ge=0)
    audio_kind_counts: dict[str, int]
    task_prefix_counts: dict[str, int]
    warning_counts: dict[str, int]
    source_hashes: dict[str, str]
    source_job_fingerprints: dict[str, str]
    source_reconcile_fingerprints: dict[str, str]
    source_stem_fingerprints: dict[str, str]
    prepared_inventory_fingerprint: str
    output_roots: dict[str, str]
    model_call_count: Literal[0] = 0
    production_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> AudioReuseFinalizationSummary:
        if self.schema_version.endswith(".2") != (self.sam_route == "resolved"):
            raise ValueError("Audio reuse finalization source contract/version differs")
        if (
            len(set(self.clip_uids)) != self.clip_count
            or len(self.clip_uids) != self.clip_count
            or self.reconcile_ready_count + self.reconcile_failed_count
            + self.upstream_unavailable_count != self.clip_count
            or len(self.unavailable_clips) != self.reconcile_failed_count + self.upstream_unavailable_count
            or self.reuse_asset_clip_count != self.reconcile_ready_count
            or self.music_asset_count != self.reuse_asset_clip_count
            or self.prepared_ready_count != self.reconcile_ready_count
            or self.prepared_failed_count != self.reconcile_failed_count
            or self.product_ready_count + self.product_failed_count != self.product_sample_count
        ):
            raise ValueError("Audio reuse finalization counts differ")
        return self


def finalize_audio_reuse_shadow(
    *, audio_production_root: Path, visual_production_root: Path, visual_runs_root: Path,
    shadow_run_id: str, case_manifest: Path, sam_route: SAMRoute | None = None,
    allow_unverified: bool = False,
) -> AudioReuseFinalizationSummary:
    """Publish new shadow outputs, preserving each existing stage's atomic writer.

    No overwrite or rollback: completed per-clip assets/stages remain intact if a
    later stage fails. Only a complete run publishes the finalization summary.
    """
    paths = jea_production_paths(audio_production_root)
    shadow = stem_shadow_root(paths.root, shadow_run_id).resolve(strict=True)
    provenance, stem_inventory, stems = validate_stem_diarization_lineage(
        shadow / "diarization", expected_shadow_root=shadow,
    )
    separation = Path(provenance.source_stem_root).resolve(strict=True)
    if provenance.route == "resolved":
        if sam_route not in (None, "voice_first"):
            raise ValueError("resolved finalizer requires fixed voice_first SAM lineage")
        sam_route = "resolved"
    else:
        sam_route = sam_route or "music_first"
    reconcile = shadow / MIMO25_STEM_RECONCILE_STAGE
    roots = {}
    for name, stage in (
        ("assets", ASSETS_STAGE), ("prepared", PREPARED_STAGE),
        ("products", PRODUCTS_STAGE), ("finalization", FINALIZATION_STAGE),
    ):
        output = shadow / stage
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        roots[name] = require_shadow_output_path(shadow_root=shadow, output_path=output)
    case_manifest = case_manifest.resolve(strict=True)
    cases = MimoCaseManifest.model_validate_json(case_manifest.read_text())
    sources = {case_manifest, paths.h3 / "samples.jsonl"}
    for stage in (separation, shadow / "diarization", shadow / "asr", reconcile):
        sources.update(stage.rglob("*.json"))
        sources.update(stage.rglob("*.jsonl"))
    hashes = {str(p): sha256_file(p) for p in sorted(sources)}
    selected_ids = set(cases.clip_uids)
    if (
        [uid for uid in stem_inventory.clip_uids if uid in selected_ids] != cases.clip_uids
        or Path(provenance.source_stem_root).resolve() != separation
        or provenance.route != sam_route
    ):
        raise ValueError("finalizer case/route/separation lineage differs")
    summary = MimoStemReconcileSummary.model_validate_json((reconcile / "summary.json").read_text())
    records = load_reconcile_sources(reconcile)
    base = build_mimo25_inventory(
        visual_production_root=visual_production_root, visual_runs_root=visual_runs_root,
        audio_production_root=paths.root, case_manifest=cases,
    )
    if (
        [j.clip_uid for j in base.jobs] != cases.clip_uids
        or base.source_h3_samples_sha256 != hashes[str(paths.h3 / "samples.jsonl")]
    ):
        raise ValueError("finalizer frozen H3 inventory differs")
    jobs = build_stem_reconcile_jobs(
        base_inventory=usable_stem_reconcile_inventory(base, provenance),
        stem_diarization_root=shadow / "diarization", stem_asr_root=shadow / "asr", route=sam_route,
    )
    if any((roots["assets"] / j.clip_uid).resolve().parent != roots["assets"] for j in jobs):
        raise ValueError("unsafe Audio reuse clip output path")
    if (
        summary.clip_uids != cases.clip_uids or summary.route != sam_route
        or summary.processed_clip_uids != [j.clip_uid for j in jobs]
        or summary.skipped_clips != [x for x in provenance.skipped_clips if x.clip_uid in selected_ids]
        or summary.diarization_failed_clips != [x for x in provenance.diarization_failed_clips if x.clip_uid in selected_ids]
    ):
        raise ValueError("finalizer reconcile inventory/lineage differs")
    selected = selected_stem_records(
        [s for s in stems if s.clip_uid in set(summary.processed_clip_uids)],
        route=sam_route, allow_unverified=allow_unverified,
    )
    samples = [FinalH3SampleV2.model_validate_json(line) for line in
               (paths.h3 / "samples.jsonl").read_text().splitlines() if line.strip()]
    values = base.model_dump(mode="json", exclude={"inventory_fingerprint"}, exclude_none=True)
    values.update(jobs=[j.model_dump(mode="json") for j in jobs], clip_count=len(jobs))
    validate_prepared_inputs(_inventory(values), records, project_prepared_samples(jobs, samples), selected)
    stems_by_clip = {s.clip_uid: s for s in selected}
    unavailable = {x.clip_uid: x.reason_code for x in [*summary.skipped_clips, *summary.diarization_failed_clips]}
    unavailable.update({r.clip_uid: r.failure_code for r in records if r.status == "failed"})
    if any(sha256_file(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError("Audio reuse finalization input changed")

    # Keep final paths stable: the existing builder atomically publishes each clip
    # and records those exact absolute paths in its immutable manifest.
    roots["assets"].mkdir()
    manifests = []
    for job, record in zip(jobs, records, strict=True):
        if record.status == "ready":
            manifests.append(build_audio_reuse_assets(
                job=job, annotation=record.annotation, stem_record=stems_by_clip[job.clip_uid],
                audio_production_root=paths.root, output_root=roots["assets"] / job.clip_uid,
                allow_unverified=allow_unverified,
            ))
    prepared = prepare_audio_reuse_sources(
        audio_production_root=paths.root, visual_production_root=visual_production_root,
        visual_runs_root=visual_runs_root, stem_shadow_root=shadow,
        base_reconcile_root=reconcile, override_reconcile_root=None,
        source_h3_root=paths.h3, prepared_root=roots["prepared"],
    )
    products = materialize_audio_reuse_products(
        mimo_root=roots["prepared"], source_h3_root=roots["prepared"] / "h3",
        separation_root=separation, reuse_root=roots["assets"],
        audio_production_root=paths.root, output_root=roots["products"],
        enable_full_audio_reuse=True,
    )
    for stage_hashes in (prepared.source_hashes, products.source_hashes):
        for path, digest in stage_hashes.items():
            if path in hashes and hashes[path] != digest:
                raise ValueError("Audio reuse stages consumed different input snapshots")
            hashes[path] = digest
    for root in (roots["prepared"], roots["products"]):
        for name in ("records.jsonl", "summary.json"):
            hashes[str(root / name)] = sha256_file(root / name)
    result = AudioReuseFinalizationSummary(
        schema_version=("r2v.h3.audio_reuse_finalization_summary.2" if sam_route == "resolved"
                        else "r2v.h3.audio_reuse_finalization_summary.1"),
        shadow_run_id=shadow_run_id, sam_route=sam_route, allow_unverified=allow_unverified,
        clip_uids=cases.clip_uids, clip_count=len(cases.clip_uids),
        reconcile_ready_count=summary.ready_count, reconcile_failed_count=summary.failed_count,
        upstream_unavailable_count=summary.skipped_clip_count, unavailable_clips=unavailable,
        reuse_asset_clip_count=len(manifests), speaker_speech_asset_count=sum(len(m.speakers) for m in manifests),
        music_asset_count=len(manifests),
        reuse_exclusion_reason_counts=dict(sorted(Counter(e.reason for m in manifests for e in m.exclusions).items())),
        prepared_ready_count=prepared.ready_count, prepared_failed_count=prepared.failed_count,
        product_sample_count=products.sample_count, product_ready_count=products.ready_count,
        product_failed_count=products.failed_count, audio_kind_counts=products.audio_kind_counts,
        task_prefix_counts=products.task_prefix_counts, warning_counts=products.warning_counts,
        source_hashes=dict(sorted(hashes.items())),
        source_job_fingerprints={j.clip_uid: j.request_fingerprint for j in jobs},
        source_reconcile_fingerprints={r.clip_uid: r.source_reconcile_record_fingerprint for r in records},
        source_stem_fingerprints={s.clip_uid: s.record_fingerprint for s in selected},
        prepared_inventory_fingerprint=prepared.inventory_fingerprint,
        output_roots={k: str(v) for k, v in roots.items()},
    )
    with tempfile.TemporaryDirectory(prefix=".reuse-finalize-", dir=shadow) as temporary:
        stage = Path(temporary) / "finalization"
        stage.mkdir()
        (stage / "summary.json").write_text(result.model_dump_json(indent=2) + "\n")
        if any(sha256_file(Path(p)) != digest for p, digest in hashes.items()):
            raise ValueError("Audio reuse finalization input changed")
        if roots["finalization"].exists():
            raise FileExistsError(roots["finalization"])
        stage.rename(roots["finalization"])
    return result
