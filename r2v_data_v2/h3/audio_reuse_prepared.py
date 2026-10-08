"""Read-only bridge from frozen stem reconcile records to Audio reuse products."""
from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import fields
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2, FinalQwen3SpeechSegment
from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoCaseManifest,
    MimoClipJob,
    MimoInventory,
    _inventory,
    build_mimo25_inventory,
    build_mimo25_reference_inventory,
    load_mimo25_reference_sources,
    project_mimo_h3_sample_references,
)
from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft, MimoBackendProvenance
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MimoStemReconcileRecord,
    MimoStemReconcileSummary,
    build_stem_reconcile_jobs,
    usable_stem_reconcile_inventory,
)
from r2v_data_v2.h3.resolved_audio_stems import (
    RESOLVED_STAGE,
    StemRecord,
    load_stem_source,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    sha256_file,
    validate_stem_diarization_lineage,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.structured_output import ValidationIssue


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FrozenReuseBackendProvenance(MimoBackendProvenance):
    # These frozen versions share the consumed annotation/runtime contract.
    schema_version: Literal["r2v.h3.mimo25_backend.60", "r2v.h3.mimo25_backend.61", "r2v.h3.mimo25_backend.62",
                            "r2v.h3.mimo25_backend.63", "r2v.h3.mimo25_backend.64",
                            "r2v.h3.mimo25_backend.65", "r2v.h3.mimo25_backend.66",
                            "r2v.h3.mimo25_backend.67", "r2v.h3.mimo25_backend.68",
                            "r2v.h3.mimo25_backend.69", "r2v.h3.mimo25_backend.70",
                            "r2v.h3.mimo25_backend.71", "r2v.h3.mimo25_backend.72",
                            "r2v.h3.mimo25_backend.73", "r2v.h3.mimo25_backend.74",
                            "r2v.h3.mimo25_backend.75", "r2v.h3.mimo25_backend.76",
                            "r2v.h3.mimo25_backend.77", "r2v.h3.mimo25_backend.78",
                            "r2v.h3.mimo25_backend.79", "r2v.h3.mimo25_backend.80"]
    prompt_version: Literal["h3_mimo25_speech_assembly_v46", "h3_mimo25_speech_assembly_v47",
                            "h3_mimo25_speech_assembly_v48", "h3_mimo25_speech_assembly_v49",
                            "h3_mimo26_ra2va_single_v1", "h3_mimo26_ra2va_single_v2",
                            "h3_mimo26_ra2va_single_v3", "h3_mimo26_ra2va_single_v4",
                            "h3_mimo26_ra2va_single_v5_compact",
                            "h3_mimo26_ra2va_single_v6_compact2",
                            "h3_mimo26_ra2va_single_v7_compact2_cleanup",
                            "h3_mimo26_ra2va_single_v8_caption_quality",
                            "h3_mimo26_ra2va_single_v9_closed_subjects",
                            "h3_mimo26_ra2va_single_v10_speaker_subject_consistency",
                            "h3_mimo26_ra2va_single_v10_dense_icl_ab_v1",
                            "h3_mimo26_ra2va_single_v10_action_icl_ab_v2",
                            "h3_mimo26_ra2va_single_v11_compact3_audio_av",
                            "h3_mimo26_ra2va_two_step_joint_v1"]
    policy_version: Literal["h3_mimo25_av_authority_contract_v17", "h3_mimo25_av_authority_contract_v18"]
    visual_prompt_version: Literal["h3_mimo25_visual_only_v4", "h3_mimo25_visual_only_v5"]


class _FrozenStemRecord(MimoStemReconcileRecord):
    backend_provenance: FrozenReuseBackendProvenance


class AudioReusePreparedSource(SchemaModel):
    schema_version: Literal["r2v.h3.audio_reuse_prepared_source.1"] = "r2v.h3.audio_reuse_prepared_source.1"
    clip_uid: str
    source_job_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_stem_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_reconcile_record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_backend_provenance: FrozenReuseBackendProvenance
    status: Literal["ready", "failed"]
    annotation: MimoAVAnnotationDraft | None
    failure_code: str | None
    failure_reason: str | None
    failure_issues: list[ValidationIssue]
    prepared_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_source(self) -> AudioReusePreparedSource:
        if self.status == "ready":
            if self.annotation is None or self.failure_code or self.failure_reason or self.failure_issues:
                raise ValueError("ready prepared source requires only finalized annotation")
        elif not self.failure_code or not self.failure_reason:
            raise ValueError("failed prepared source requires original failure provenance")
        if self.prepared_fingerprint != _hash(self.model_dump(mode="json", exclude={"prepared_fingerprint"})):
            raise ValueError("prepared source fingerprint mismatch")
        return self


class AudioReusePreparedSummary(SchemaModel):
    schema_version: Literal["r2v.h3.audio_reuse_prepared_source_summary.1"] = "r2v.h3.audio_reuse_prepared_source_summary.1"
    clip_uids: list[str]
    ready_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    source_hashes: dict[str, str]
    chosen_reconcile_roots: dict[str, str]
    inventory_fingerprint: str
    prepared_records_sha256: str
    prepared_samples_sha256: str
    model_call_count: Literal[0] = 0
    production_artifacts_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> AudioReusePreparedSummary:
        if len(set(self.clip_uids)) != len(self.clip_uids) or self.ready_count + self.failed_count != len(self.clip_uids):
            raise ValueError("prepared source counts differ")
        if set(self.chosen_reconcile_roots) != set(self.clip_uids):
            raise ValueError("prepared source origins differ")
        return self


class FrozenAudioReuseInventory(SchemaModel):
    """Prepared jobs from reconcile, without claiming a rebuilt Visual inventory."""

    schema_version: Literal["r2v.h3.audio_reuse_frozen_inventory.1"] = "r2v.h3.audio_reuse_frozen_inventory.1"
    source_reconcile_root: str
    source_stem_root: str
    source_h3_root: str
    source_h3_samples_sha256: str
    clip_count: int
    jobs: list[MimoClipJob]
    inventory_fingerprint: str

    @model_validator(mode="after")
    def validate_inventory(self) -> FrozenAudioReuseInventory:
        if self.clip_count != len(self.jobs) or len({j.clip_uid for j in self.jobs}) != self.clip_count:
            raise ValueError("frozen prepared job inventory differs")
        if self.inventory_fingerprint != _hash(self.model_dump(mode="json", exclude={"inventory_fingerprint"})):
            raise ValueError("frozen prepared inventory fingerprint mismatch")
        return self


def load_prepared_inventory(path: Path) -> MimoInventory | FrozenAudioReuseInventory:
    values = json.loads(path.read_text())
    model = (FrozenAudioReuseInventory if values.get("schema_version") == "r2v.h3.audio_reuse_frozen_inventory.1"
             else MimoInventory)
    return model.model_validate(values)


def _publish_prepared(
    output: Path, inventory: MimoInventory | FrozenAudioReuseInventory,
    summary: AudioReusePreparedSummary, samples_text: str, records_text: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reuse-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        (stage / "h3").mkdir(parents=True)
        (stage / "h3/samples.jsonl").write_text(samples_text)
        (stage / "inventory.json").write_text(inventory.model_dump_json(indent=2) + "\n")
        (stage / "records.jsonl").write_text(records_text)
        (stage / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
        if any(sha256_file(Path(path)) != digest for path, digest in summary.source_hashes.items()):
            raise ValueError("frozen preparation input changed")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)


def load_reconcile_sources(root: Path) -> list[AudioReusePreparedSource]:
    """Validate real stem records, including their original raw/config hashes."""
    result = []
    for line in (root / "records.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        record = _FrozenStemRecord.model_validate(raw)
        if record.record_fingerprint != _hash({k: v for k, v in raw.items() if k != "record_fingerprint"}):
            raise ValueError("frozen reconcile raw fingerprint mismatch")
        values = {
            "schema_version": "r2v.h3.audio_reuse_prepared_source.1",
            "clip_uid": record.clip_uid,
            "source_job_fingerprint": record.source_job_fingerprint,
            "source_stem_record_fingerprint": record.source_stem_record_fingerprint,
            "source_reconcile_record_fingerprint": record.record_fingerprint,
            "source_backend_provenance": raw["backend_provenance"],
            "status": record.status, "annotation": raw["annotation"],
            "failure_code": record.failure_code, "failure_reason": record.failure_reason,
            "failure_issues": raw["failure_issues"],
        }
        result.append(AudioReusePreparedSource(**values, prepared_fingerprint=_hash(values)))
    _unique(result)
    summary_path = root / "summary.json"
    if summary_path.is_file():
        summary = MimoStemReconcileSummary.model_validate_json(summary_path.read_text())
        counts = Counter(r.status for r in result)
        if (summary.processed_clip_uids != [r.clip_uid for r in result]
                or summary.ready_count != counts["ready"] or summary.failed_count != counts["failed"]):
            raise ValueError("frozen reconcile summary differs from records")
    return result


def _unique(records: list) -> dict:
    indexed = {r.clip_uid: r for r in records}
    if len(indexed) != len(records):
        raise ValueError("duplicate prepared/reconcile clip ID")
    return indexed


def overlay_reconcile_sources(
    base: list[AudioReusePreparedSource], override: list[AudioReusePreparedSource],
) -> list[AudioReusePreparedSource]:
    by_clip, replacements = _unique(base), _unique(override)
    if not set(replacements) <= set(by_clip):
        raise ValueError("override contains unknown clip")
    for uid, replacement in replacements.items():
        original = by_clip[uid]
        if replacement.source_job_fingerprint != original.source_job_fingerprint:
            raise ValueError("override source job fingerprint mismatch")
        if replacement.source_stem_record_fingerprint != original.source_stem_record_fingerprint:
            raise ValueError("override source stem fingerprint mismatch")
    return [replacements.get(r.clip_uid, r) for r in base]


def validate_prepared_inputs(
    inventory: MimoInventory | FrozenAudioReuseInventory, records: list[AudioReusePreparedSource],
    samples: list[FinalH3SampleV2], stems: list[StemRecord],
) -> None:
    if [r.clip_uid for r in records] != [j.clip_uid for j in inventory.jobs]:
        raise ValueError("prepared records differ from ordered job inventory")
    stem_by_clip = _unique(stems)
    sample_ids = [s.sample_id for s in samples]
    if len(set(sample_ids)) != len(sample_ids) or {s.clip_uid for s in samples} != {j.clip_uid for j in inventory.jobs}:
        raise ValueError("prepared H3 sample inventory differs")
    for job, record in zip(inventory.jobs, records, strict=True):
        stem = stem_by_clip.get(job.clip_uid)
        if record.source_job_fingerprint != job.request_fingerprint:
            raise ValueError("prepared source job fingerprint mismatch")
        if stem is None or record.source_stem_record_fingerprint != stem.record_fingerprint:
            raise ValueError("prepared source SAM fingerprint mismatch")
        if sorted(s.sample_id for s in samples if s.clip_uid == job.clip_uid) != sorted(job.source_h3_sample_ids):
            raise ValueError("prepared source H3 sample IDs differ")
    if project_prepared_samples(inventory.jobs, samples) != samples:
        raise ValueError("prepared speech differs from exact current job inventory")


def project_prepared_samples(jobs: list[MimoClipJob], samples: list[FinalH3SampleV2]) -> list[FinalH3SampleV2]:
    by_clip = _unique(jobs)
    projected = []
    for sample in samples:
        if sample.clip_uid not in by_clip:
            continue
        job = by_clip[sample.clip_uid]
        if (sample.target_video != job.target_video_path or sample.target_full_audio_path != job.target_full_audio_path
                or sample.target_full_audio_sha256 != job.target_full_audio_sha256):
            raise ValueError("prepared H3 target media differs from reconstructed job")
        speech = [
            FinalQwen3SpeechSegment(
                segment_id=s.segment_id, speaker_cluster_id=s.source_speaker_cluster_id,
                source_start_sample=s.source_start_sample, source_end_sample=s.source_end_sample,
                source_sample_rate_hz=s.source_sample_rate_hz, start_time=s.start_time, end_time=s.end_time,
                text=s.asr_text, language=s.asr_language,
            ) for s in job.segments if s.asr_status == "transcribed"
        ]
        values = sample.model_dump(mode="json")
        values["speech_segments"] = [s.model_dump(mode="json") for s in speech]
        projected.append(FinalH3SampleV2.model_validate(values))
    return projected


def prepare_frozen_audio_reuse_sources(
    *, reconcile_root: Path, prepared_root: Path, clip_uids: list[str] | None = None,
    source_h3_root: Path | None = None,
) -> AudioReusePreparedSummary:
    """Consume frozen jobs/annotations; never rebuild Visual or diarization inputs."""
    reconcile = reconcile_root.expanduser().resolve(strict=True)
    output = prepared_root.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    contract_path = reconcile / "source_contract.json"
    contract = json.loads(contract_path.read_text())
    if contract.get("binding_evidence_mode") != "none":
        raise ValueError("frozen preparation requires reference-only source jobs")
    jobs = [MimoClipJob.model_validate(row) for row in contract["jobs"]]
    records = load_reconcile_sources(reconcile)
    by_clip = _unique(records)
    requested = set(clip_uids) if clip_uids is not None else set(by_clip)
    if requested - set(by_clip):
        raise ValueError("unknown frozen reconcile clip UID")
    jobs = [j for j in jobs if j.clip_uid in requested and by_clip[j.clip_uid].status == "ready"]
    if not jobs:
        raise ValueError("no ready frozen reconcile clips selected")
    records = [by_clip[j.clip_uid] for j in jobs]
    stem_root = reconcile.parent / RESOLVED_STAGE
    stem_inventory, stems, _ = load_stem_source(stem_root)
    audio_root = Path(stem_inventory.source_canonical_audio_manifest_path).parent.parent
    h3 = (source_h3_root or audio_root / "h3").expanduser().resolve()
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in (reconcile, stem_root, h3)):
        raise ValueError("frozen prepared output overlaps source stages")
    samples_path = h3 / "samples.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen canonical H3 sample metadata: {samples_path}; "
            "supply --source-h3-root with the matching canonical samples.jsonl "
            "(sample identity and original visual references are required)"
        )
    ids = {sample_id for job in jobs for sample_id in job.source_h3_sample_ids}
    canonical = {}
    with samples_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("sample_id") not in ids or row.get("pair_type") != "canonical":
                continue
            # Frozen canonical metadata supplies identity, not legacy voice binding.
            row["subject_voices"] = []
            sample = FinalH3SampleV2.model_validate(row)
            if sample.sample_id in canonical:
                raise ValueError("duplicate frozen canonical H3 sample")
            canonical[sample.sample_id] = sample
    if ids != set(canonical):
        raise ValueError(f"missing frozen canonical H3 samples in {samples_path}: {sorted(ids - set(canonical))}")
    from r2v_data_v2.h3.qwen38_h3_recaption import build_reference_contract

    samples = []
    for job in jobs:
        if job.binding_evidence_mode != "none" or len(job.source_h3_sample_ids) != 1:
            raise ValueError("frozen reference-only job requires one canonical source sample")
        sample = canonical[job.source_h3_sample_ids[0]].model_copy(update={"r2v_instruction": job.r2v_instruction})
        projected = project_mimo_h3_sample_references(
            sample, reference_images=job.reference_images,
            reference_selection=job.reference_selection,
        )
        if build_reference_contract(projected, "visual_only").subjects != job.reference_subjects:
            raise ValueError("frozen H3 Subject graph differs from reconcile job")
        # The materializer needs the complete canonical inventory for its own projection.
        samples.append(sample)
    samples = project_prepared_samples(jobs, samples)
    samples_text = "".join(s.model_dump_json() + "\n" for s in samples)
    values = {
        "schema_version": "r2v.h3.audio_reuse_frozen_inventory.1",
        "source_reconcile_root": str(reconcile), "source_stem_root": str(stem_root), "source_h3_root": str(h3),
        "source_h3_samples_sha256": hashlib.sha256(samples_text.encode()).hexdigest(),
        "clip_count": len(jobs), "jobs": [j.model_dump(mode="json") for j in jobs],
    }
    inventory = FrozenAudioReuseInventory(**values, inventory_fingerprint=_hash(values))
    validate_prepared_inputs(inventory, records, samples, stems)
    records_text = "".join(r.model_dump_json() + "\n" for r in records)
    inputs = [contract_path, reconcile / "records.jsonl", reconcile / "summary.json",
              samples_path, stem_root / "records.jsonl"]
    summary = AudioReusePreparedSummary(
        clip_uids=[j.clip_uid for j in jobs], ready_count=len(jobs), failed_count=0, sample_count=len(samples),
        source_hashes={str(p): sha256_file(p) for p in inputs},
        chosen_reconcile_roots={j.clip_uid: str(reconcile) for j in jobs},
        inventory_fingerprint=inventory.inventory_fingerprint,
        prepared_records_sha256=hashlib.sha256(records_text.encode()).hexdigest(),
        prepared_samples_sha256=inventory.source_h3_samples_sha256,
    )
    _publish_prepared(output, inventory, summary, samples_text, records_text)
    return summary


def prepare_audio_reuse_sources(
    *, audio_production_root: Path, visual_production_root: Path, visual_runs_root: Path,
    stem_shadow_root: Path, base_reconcile_root: Path, override_reconcile_root: Path | None,
    source_h3_root: Path | None = None, prepared_root: Path,
    binding_evidence_mode: Literal["legacy_lr_asd", "none"] = "legacy_lr_asd",
    stem_diarization_root: Path | None = None, stem_asr_root: Path | None = None,
) -> AudioReusePreparedSummary:
    """Publish a NEW prepared shadow root without invoking any model or media writer."""
    output = prepared_root.resolve()
    paths = jea_production_paths(audio_production_root)
    shadow = stem_shadow_root.resolve(strict=True)
    from r2v_data_v2.h3.sam_audio_stem_shadow import require_shadow_output_path

    diarization = require_shadow_output_path(
        shadow_root=shadow, output_path=stem_diarization_root or shadow / "diarization",
    )
    asr = require_shadow_output_path(
        shadow_root=shadow, output_path=stem_asr_root or shadow / "asr",
    )
    stem_sources = [shadow / n for n in ("separation", "auk_speech_v1", "resolved_stems_v1")]
    stem_sources += [diarization, asr]
    inputs = [base_reconcile_root.resolve(strict=True)]
    if binding_evidence_mode == "legacy_lr_asd":
        if source_h3_root is None:
            raise ValueError("legacy preparation requires source_h3_root")
        inputs.append(source_h3_root.resolve(strict=True))
    if override_reconcile_root is not None:
        inputs.append(override_reconcile_root.resolve(strict=True))
    protected = [getattr(paths, f.name).resolve() for f in fields(paths) if f.name != "root"]
    protected += inputs + stem_sources
    protected += [visual_production_root.resolve(), visual_runs_root.resolve()]
    if paths.root.is_relative_to(output) or shadow.is_relative_to(output) or any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in protected
    ):
        raise ValueError("prepared output overlaps protected inputs/stages")
    if output.exists():
        raise FileExistsError(output)
    # Capture durable inputs before reconstruction and verify again before publication.
    source_files = set()
    for root in inputs + stem_sources:
        if root.is_dir():
            source_files.update(p for p in root.rglob("*.json") if p.is_file())
            source_files.update(p for p in root.rglob("*.jsonl") if p.is_file())
    source_files.update([
        visual_production_root / "samples.jsonl", paths.audio / "canonical_clips.jsonl",
    ])
    if binding_evidence_mode == "legacy_lr_asd":
        source_files.update([
        paths.diarization / "inventory.json", paths.diarization / "raw_segments.jsonl",
        paths.diarization / "bound_segments.jsonl", paths.asr / "segments.jsonl",
        paths.root / "binding_audit_v1/segments.jsonl", paths.h3 / "samples.jsonl",
        ])
    elif binding_evidence_mode == "none":
        source_files = {p for p in source_files if p.name not in {"bound_segments.jsonl", "cluster_bindings.jsonl"}}
    else:
        raise ValueError("invalid prepared binding evidence mode")
    hashes = {str(p): sha256_file(p) for p in sorted(source_files)}
    base = load_reconcile_sources(base_reconcile_root)
    override = load_reconcile_sources(override_reconcile_root) if override_reconcile_root else []
    effective = overlay_reconcile_sources(base, override)
    manifest = MimoCaseManifest(clip_uids=[r.clip_uid for r in base])
    provenance, stem_inventory, stems = validate_stem_diarization_lineage(
        diarization, expected_shadow_root=shadow,
    )
    if [uid for uid in stem_inventory.clip_uids if uid in set(manifest.clip_uids)] != manifest.clip_uids:
        raise ValueError("prepared clips are not an ordered separation subset")
    builder = build_mimo25_reference_inventory if binding_evidence_mode == "none" else build_mimo25_inventory
    base_inventory = builder(
        visual_production_root=visual_production_root, visual_runs_root=visual_runs_root,
        audio_production_root=audio_production_root, case_manifest=manifest,
        **({"verify_video": False, "verify_audio": False} if binding_evidence_mode == "none" else {}),
    )
    if (binding_evidence_mode == "legacy_lr_asd"
        and sha256_file(source_h3_root / "samples.jsonl") != base_inventory.source_h3_samples_sha256):
        raise ValueError("frozen H3 source differs from reconstruction source")
    jobs = build_stem_reconcile_jobs(
        base_inventory=usable_stem_reconcile_inventory(base_inventory, provenance),
        stem_diarization_root=diarization, stem_asr_root=asr, route=provenance.route,
        binding_evidence_mode=binding_evidence_mode,
    )
    if binding_evidence_mode == "none":
        source_samples, _ = load_mimo25_reference_sources(
            visual_production_root=visual_production_root, visual_runs_root=visual_runs_root,
            audio_production_root=audio_production_root,
            verify_video=False, verify_audio=False,
        )
        source_text = "".join(s.model_dump_json() + "\n" for s in source_samples)
        if hashlib.sha256(source_text.encode()).hexdigest() != base_inventory.source_h3_samples_sha256:
            raise ValueError("shadow canonical source changed during reconstruction")
    else:
        source_samples = [
            FinalH3SampleV2.model_validate_json(line)
            for line in (source_h3_root / "samples.jsonl").read_text().splitlines() if line.strip()
        ]
    samples = project_prepared_samples(jobs, source_samples)
    values = base_inventory.model_dump(mode="json", exclude={"inventory_fingerprint"})
    if values["source_diarization_inventory_sha256"] is None:
        values.pop("source_diarization_inventory_sha256")
    values.update(jobs=[j.model_dump(mode="json") for j in jobs], clip_count=len(jobs))
    for name, path in (
        ("source_diarization_inventory_sha256", diarization / "inventory.json"),
        ("source_diarization_raw_segments_sha256", diarization / "raw_segments.jsonl"),
        ("source_diarization_bound_segments_sha256", diarization / "bound_segments.jsonl"),
        ("source_qwen3_asr_segments_sha256", asr / "segments.jsonl"),
    ):
        if name == "source_diarization_bound_segments_sha256" and binding_evidence_mode == "none":
            continue
        values[name] = sha256_file(path)
    # The outer inventory binds the new H3 file; the reconstructed jobs are unchanged.
    samples_text = "".join(s.model_dump_json() + "\n" for s in samples)
    values["source_h3_samples_sha256"] = hashlib.sha256(samples_text.encode()).hexdigest()
    inventory = _inventory(values)
    selected_stems = [s for s in stems if s.route == provenance.route]
    validate_prepared_inputs(inventory, effective, samples, selected_stems)
    records_text = "".join(r.model_dump_json() + "\n" for r in effective)
    counts = Counter(r.status for r in effective)
    overridden = {r.clip_uid for r in override}
    summary = AudioReusePreparedSummary(
        clip_uids=manifest.clip_uids, ready_count=counts["ready"], failed_count=counts["failed"],
        sample_count=len(samples), source_hashes=dict(sorted(hashes.items())),
        chosen_reconcile_roots={r.clip_uid: str((override_reconcile_root if r.clip_uid in overridden else base_reconcile_root).resolve()) for r in effective},
        inventory_fingerprint=inventory.inventory_fingerprint,
        prepared_records_sha256=hashlib.sha256(records_text.encode()).hexdigest(),
        prepared_samples_sha256=values["source_h3_samples_sha256"],
    )
    _publish_prepared(output, inventory, summary, samples_text, records_text)
    return summary
