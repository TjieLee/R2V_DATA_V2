"""Read-only discovery and ordering of current-run H3 review variants."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from r2v_data_v2.h3.audio_reuse import AudioReuseManifest
from r2v_data_v2.h3.audio_reuse_materializer import (
    AUDIO_REUSE_MATERIALIZER_VERSION,
    AudioReuseProduct,
    AudioReuseProductSummary,
)
from r2v_data_v2.h3.audio_reuse_prepared import (
    AudioReusePreparedSource,
    project_prepared_samples,
    validate_prepared_inputs,
)
from r2v_data_v2.h3.frame_conditioning_projection import (
    FRAME_STAGE,
    VISUAL_TASKS,
    load_frame_conditioned_products,
)
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.mimo25_av_reconcile import MimoInventory, MimoRecord
from r2v_data_v2.h3.mimo25_backend import MIMO25_MATERIALIZER_VERSION
from r2v_data_v2.h3.mimo25_h3_materializer import (
    MimoH3ShadowRecord,
    MimoH3ShadowSummary,
)
from r2v_data_v2.h3.mimo25_stem_shadow import MIMO25_STEM_RECONCILE_STAGE
from r2v_data_v2.h3.sam_audio_stem_shadow import sha256_file

VARIANT_TYPES = (
    "visual_only", "target_voice_reference", "cross_voice_reference", "full_audio_reuse",
    "music_reference", "speaker_speech_reuse", "music_reuse",
)
_PRIORITY = {"audio_reuse_product": 0, "frame_conditioned_product": 0, "mimo_h3_shadow": 1, "source_h3": 2, "qa_derived": 3}


def finalize_registry(variants: list[dict], fingerprint: Callable) -> tuple[list[dict], dict]:
    """Deduplicate only equivalent contracts/media/text, preferring publication."""
    unique = {}
    published_keys = set()
    for variant in sorted(variants, key=lambda v: (_PRIORITY[v["source_kind"]], v["sample_id"], v.get("source_root", ""))):
        mode = variant.setdefault("visual_reference_mode", "reference")
        variant["visual_task"] = VISUAL_TASKS[mode]
        final = variant["source_kind"] in {"audio_reuse_product", "frame_conditioned_product"} and variant["status"] == "ready"
        variant["review_family"] = "final_4way" if final else "legacy_other"
        variant.setdefault("source_product_sample_id", variant["sample_id"] if final else None)
        refs = variant.get("references") or {}
        key = fingerprint({
            "visual_reference_mode": mode,
            "conditioning_variant": variant["conditioning_variant"],
            "pictures": refs.get("pictures"), "subjects": refs.get("subjects"),
            "audios": [{k: v for k, v in a.items() if k not in {"provenance", "role", "source_type", "url"}}
                       for a in refs.get("audios", [])],
            "text": variant.get("text"), "status": variant["status"],
            # Distinct unavailable variants must not erase one another.
            "failure": variant.get("issues", variant.get("reason")),
        })
        if final:
            published_keys.add(key)
        elif key in published_keys:
            continue
        unique.setdefault((key, variant["source_product_sample_id"] if final else None), variant)
    result = sorted(unique.values(), key=lambda v: (
        0 if v["review_family"] == "final_4way" else 1,
        v["source_product_sample_id"] or "",
        list(VISUAL_TASKS).index(v["visual_reference_mode"]),
        0 if v["conditioning_variant"] == "visual_only" else 1 if v["source_kind"] == "source_h3" else 2,
        v["sample_id"], _PRIORITY[v["source_kind"]], v.get("source_root", ""),
    ))
    coverage = {}
    for kind in VARIANT_TYPES:
        matching = [v for v in result if v["status"] == "ready" and (
            v["conditioning_variant"] == kind or any(
                a["kind"] == kind for a in (v.get("references") or {}).get("audios", [])
            )
        )]
        coverage[kind] = ("available" if any(v["publication_status"] == "published" for v in matching)
                          or any(v["source_kind"] == "source_h3" for v in matching)
                          else "review-only available" if matching else "not generated")
    return result, coverage


def discover_products(*, shadow: Path, jobs: list, reconcile: list, samples: list,
                      stems: list, inventories: list[MimoInventory], source_hashes: dict,
                      fingerprint: Callable) -> list[tuple[str, Path, object, FinalH3SampleV2]]:
    """Inspect only immediate current-run stage summaries, never search other runs."""
    by_job = {j.clip_uid: j for j in jobs}
    by_record = {r.clip_uid: r for r in reconcile}
    by_stem = {s.clip_uid: s for s in stems}
    original = {s.sample_id: s for s in samples}
    prepared = {s.sample_id: s for s in project_prepared_samples(jobs, samples)}

    def read(path, model):
        path = Path(path).resolve(strict=True)
        source_hashes[str(path)] = sha256_file(path)
        return model.model_validate_json(path.read_text())

    def rows(path, model):
        path = Path(path).resolve(strict=True)
        source_hashes[str(path)] = sha256_file(path)
        return [model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]

    result = []
    stage_roots = sorted(p for p in shadow.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name != "qa")
    for root in stage_roots:
        if not root.is_dir() or root.name.startswith(".") or root.name == "qa":
            continue
        summary_path = root / "summary.json"
        known = root.name in {"h3_audio_reuse_products_v1", "mimo25_h3_shadow_v5", FRAME_STAGE}
        if not summary_path.exists():
            if known and (root / "records.jsonl").exists():
                raise ValueError("QA published stage is missing its summary")
            continue
        if not root.resolve().is_relative_to(shadow.resolve()):
            raise ValueError("QA optional stage escapes current shadow run")
        try:
            schema = json.loads(summary_path.read_text()).get("schema_version")
        except (ValueError, AttributeError):
            if known:
                raise ValueError("QA published stage has invalid summary") from None
            continue
        if schema == "r2v.h3.audio_reuse_product_summary.1":
            source_fingerprints = {uid: r.record_fingerprint for uid, r in by_record.items()}
            summary = read(summary_path, AudioReuseProductSummary)
            records = rows(root / "records.jsonl", AudioReuseProduct)
            if summary.materializer_version != AUDIO_REUSE_MATERIALIZER_VERSION:
                raise ValueError("QA reuse materializer provenance is stale")
            if str(shadow / "separation/records.jsonl") not in summary.source_hashes:
                raise ValueError("QA reuse separation lineage is not current run")
            # The publisher records every dependency, including the prepared snapshot.
            for path, digest in summary.source_hashes.items():
                if sha256_file(Path(path)) != digest:
                    raise ValueError("QA reuse source hash mismatch")
                source_hashes[path] = digest
            inv_paths = [Path(p) for p in summary.source_hashes if Path(p).name == "inventory.json"]
            sample_paths = [Path(p) for p in summary.source_hashes if Path(p).name == "samples.jsonl"]
            if len(inv_paths) != 1 or len(sample_paths) != 1:
                raise ValueError("QA reuse prepared source provenance is ambiguous")
            inventory = read(inv_paths[0], MimoInventory)
            prepared_records_path = inv_paths[0].parent / "records.jsonl"
            if str(prepared_records_path) not in summary.source_hashes:
                raise ValueError("QA reuse prepared records hash missing")
            prepared_records = rows(prepared_records_path, AudioReusePreparedSource)
            product_samples = rows(sample_paths[0], FinalH3SampleV2)
            if inventory.source_h3_samples_sha256 != sha256_file(sample_paths[0]):
                raise ValueError("QA reuse source sample inventory hash mismatch")
            validate_prepared_inputs(inventory, prepared_records, product_samples, stems)
            if (summary.clip_uids != [j.clip_uid for j in inventory.jobs]
                    or any(j.clip_uid not in by_job or j.request_fingerprint != by_job[j.clip_uid].request_fingerprint
                           for j in inventory.jobs)
                    or any(r.source_reconcile_record_fingerprint != by_record[r.clip_uid].record_fingerprint
                           for r in prepared_records)):
                raise ValueError("QA reuse product differs from current run")
            source_samples = {s.sample_id: s for s in product_samples}
            if any(s != prepared.get(s.sample_id) for s in product_samples):
                raise ValueError("QA reuse source H3 differs from current frozen samples")
            kind = "audio_reuse_product"
        elif schema == "r2v.h3.mimo25_h3_shadow_summary.17":
            summary = read(summary_path, MimoH3ShadowSummary)
            records = rows(root / "records.jsonl", MimoH3ShadowRecord)
            source_fingerprints = {uid: r.record_fingerprint for uid, r in by_record.items()}
            current_inventories = [i for i in inventories if [j.request_fingerprint for j in i.jobs]
                                   == [j.request_fingerprint for j in jobs]]
            source_samples = original
            direct_current = (
                summary.source_mimo_inventory_fingerprint in {i.inventory_fingerprint for i in current_inventories}
                and summary.source_mimo_records_sha256 == sha256_file(shadow / MIMO25_STEM_RECONCILE_STAGE / "records.jsonl")
                and summary.source_h3_samples_sha256 in {i.source_h3_samples_sha256 for i in current_inventories}
            )
            if not direct_current:
                # The standalone publisher uses MimoRecord, not a stem-record wrapper.
                matches = []
                for stage in stage_roots:
                    path = stage / "inventory.json"
                    if path.is_file() and json.loads(path.read_text()).get("inventory_fingerprint") == summary.source_mimo_inventory_fingerprint:
                        matches.append(stage)
                if len(matches) != 1 or not matches[0].resolve().is_relative_to(shadow.resolve()):
                    raise ValueError("QA MiMo shadow source inventory is missing/ambiguous")
                stage = matches[0]
                source_inventory = read(stage / "inventory.json", MimoInventory)
                source_records = rows(stage / "records.jsonl", MimoRecord)
                if (sha256_file(stage / "records.jsonl") != summary.source_mimo_records_sha256
                        or source_inventory.source_h3_samples_sha256 != summary.source_h3_samples_sha256
                        or [r.clip_uid for r in source_records] != [j.clip_uid for j in source_inventory.jobs]
                        or any(j.clip_uid not in by_job or j.request_fingerprint != by_job[j.clip_uid].request_fingerprint
                               for j in source_inventory.jobs)
                        or any(r.inventory_fingerprint != source_inventory.inventory_fingerprint
                               or r.request_fingerprint != by_job[r.clip_uid].request_fingerprint
                               or r.annotation != by_record[r.clip_uid].annotation
                               or r.status != by_record[r.clip_uid].status for r in source_records)):
                    raise ValueError("QA MiMo shadow differs from current run provenance")
                source_fingerprints = {r.clip_uid: r.record_fingerprint for r in source_records}
                if summary.source_h3_samples_sha256 not in {i.source_h3_samples_sha256 for i in inventories}:
                    paths = [p for stage in stage_roots for p in (stage / "samples.jsonl", stage / "h3/samples.jsonl")
                             if p.is_file() and sha256_file(p) == summary.source_h3_samples_sha256]
                    if len(paths) != 1:
                        raise ValueError("QA MiMo source H3 snapshot is missing/ambiguous")
                    source_samples = {s.sample_id: s for s in rows(paths[0], FinalH3SampleV2)}
                    if any(s != prepared.get(s.sample_id) for s in source_samples.values()):
                        raise ValueError("QA MiMo source H3 differs from frozen current samples")
            kind = "mimo_h3_shadow"
        elif schema == "r2v.h3.frame_conditioned_product_summary.1":
            continue  # Load after the source Audio products have been validated.
        elif known:
            raise ValueError("QA published stage schema is not recognized")
        else:
            continue
        if (len({r.sample_id for r in records}) != len(records)
                or summary.sample_count != len(records)
                or summary.ready_count != sum(r.status == "ready" for r in records)
                or summary.failed_count != sum(r.status == "failed" for r in records)):
            raise ValueError("QA published variant inventory/count mismatch")
        for record in records:
            sample = source_samples.get(record.source_h3_sample_id)
            if (sample is None or record.clip_uid != sample.clip_uid or record.clip_uid not in by_job
                    or record.source_h3_sample_id not in by_job[record.clip_uid].source_h3_sample_ids
                    or record.source_h3_sample_sha256 != fingerprint(sample.model_dump(mode="json"))
                    or record.source_mimo_record_fingerprint != source_fingerprints.get(record.clip_uid)
                    or (record.status == "ready" and by_record[record.clip_uid].status != "ready")
                    or record.materializer_version != (AUDIO_REUSE_MATERIALIZER_VERSION if kind == "audio_reuse_product"
                                                        else MIMO25_MATERIALIZER_VERSION)):
                raise ValueError("QA published variant has stale source provenance")
            if kind == "audio_reuse_product":
                for ref in record.audio_references:
                    origin = by_record.get(ref.source_clip_uid)
                    if origin is None or ref.source_mimo_record_fingerprint != origin.record_fingerprint:
                        raise ValueError("QA reuse Audio source/donor is stale")
                    if ref.reuse_manifest_path:
                        if sha256_file(Path(ref.reuse_manifest_path)) != ref.reuse_manifest_sha256:
                            raise ValueError("QA reuse manifest hash mismatch")
                        manifest = read(ref.reuse_manifest_path, AudioReuseManifest)
                        if (manifest.clip_uid != ref.source_clip_uid
                                or fingerprint(manifest.model_dump(mode="json")) != ref.reuse_manifest_fingerprint
                                or manifest.source_job_fingerprint != by_job[ref.source_clip_uid].request_fingerprint
                                or manifest.source_annotation_sha256 != fingerprint(origin.annotation.model_dump(mode="json"))
                                or manifest.source_stem_record_fingerprint != by_stem[ref.source_clip_uid].record_fingerprint
                                or ref.reuse_asset not in [*manifest.speakers, manifest.music]):
                            raise ValueError("QA reuse asset manifest lineage mismatch")
                    if ref.reuse_asset:
                        for path, digest in ((ref.reuse_asset.source_stem_path, ref.reuse_asset.source_stem_sha256),
                                             (ref.reuse_asset.output_path, ref.reuse_asset.output_sha256)):
                            if sha256_file(Path(path)) != digest:
                                raise ValueError("QA reuse media source hash mismatch")
                            source_hashes[path] = digest
            else:
                for recovery in record.recovered_voice_references:
                    if (recovery.clip_uid != record.clip_uid
                            or recovery.mimo_record_fingerprint != record.source_mimo_record_fingerprint):
                        raise ValueError("QA recovered voice source provenance is stale")
                    if recovery.status == "selected" and not any(
                        a.source_segment_id == recovery.source_segment_id and a.entity_id == recovery.entity_id
                        and a.audio_path == recovery.output_path and a.audio_sha256 == recovery.output_sha256
                        for a in record.audio_references
                    ):
                        raise ValueError("QA recovered voice differs from published Audio")
            result.append((kind, root, record, sample))
    audio_products = {record.sample_id: (record, sample) for kind, _, record, sample in result if kind == "audio_reuse_product"}
    frame_keys = set()
    for root in stage_roots:
        path = root / "summary.json"
        if not path.is_file():
            continue
        try:
            schema = json.loads(path.read_text()).get("schema_version")
        except (ValueError, AttributeError):
            continue
        if schema != "r2v.h3.frame_conditioned_product_summary.1":
            continue
        projected, dependencies = load_frame_conditioned_products(root, shadow)
        source_hashes.update(dependencies)
        for record in projected:
            key = (record.source_product_sample_id, record.visual_reference_mode)
            if key in frame_keys:
                raise ValueError("QA frame projection source/mode is ambiguous across stages")
            frame_keys.add(key)
            source = audio_products.get(record.source_product_sample_id)
            if (source is None or source[0].status != "ready"
                    or source[0].record_fingerprint != record.source_product_record_fingerprint
                    or source[0].audio_references != record.audio_references
                    or source[0].corrected_speech_segments != record.corrected_speech_segments):
                raise ValueError("QA frame projection is not bound to current published Audio product")
            result.append(("frame_conditioned_product", root, record, source[1]))
    return result
