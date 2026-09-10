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
)
from r2v_data_v2.h3.mimo25_backend import MimoAVAnnotationDraft, MimoBackendProvenance
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MimoStemReconcileRecord,
    MimoStemReconcileSummary,
    build_stem_reconcile_jobs,
    usable_stem_reconcile_inventory,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    SAMAudioStemRecord,
    load_stem_shadow,
    sha256_file,
    validate_stem_diarization_lineage,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.structured_output import ValidationIssue


def _hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FrozenReuseBackendProvenance(MimoBackendProvenance):
    # These frozen versions share the consumed annotation/runtime contract.
    schema_version: Literal["r2v.h3.mimo25_backend.60", "r2v.h3.mimo25_backend.61", "r2v.h3.mimo25_backend.62"]
    prompt_version: Literal["h3_mimo25_speech_assembly_v46", "h3_mimo25_speech_assembly_v47"]


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
    inventory: MimoInventory, records: list[AudioReusePreparedSource],
    samples: list[FinalH3SampleV2], stems: list[SAMAudioStemRecord],
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


def prepare_audio_reuse_sources(
    *, audio_production_root: Path, visual_production_root: Path, visual_runs_root: Path,
    stem_shadow_root: Path, base_reconcile_root: Path, override_reconcile_root: Path | None,
    source_h3_root: Path, prepared_root: Path,
) -> AudioReusePreparedSummary:
    """Publish a NEW prepared shadow root without invoking any model or media writer."""
    output = prepared_root.resolve()
    paths = jea_production_paths(audio_production_root)
    shadow = stem_shadow_root.resolve(strict=True)
    inputs = [base_reconcile_root.resolve(strict=True), source_h3_root.resolve(strict=True)]
    if override_reconcile_root is not None:
        inputs.append(override_reconcile_root.resolve(strict=True))
    protected = [getattr(paths, f.name).resolve() for f in fields(paths) if f.name != "root"]
    protected += inputs + [shadow / n for n in ("separation", "diarization", "asr")]
    protected += [visual_production_root.resolve(), visual_runs_root.resolve()]
    if paths.root.is_relative_to(output) or shadow.is_relative_to(output) or any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in protected
    ):
        raise ValueError("prepared output overlaps protected inputs/stages")
    if output.exists():
        raise FileExistsError(output)
    # Capture durable inputs before reconstruction and verify again before publication.
    source_files = set()
    for root in inputs + [shadow / n for n in ("separation", "diarization", "asr")]:
        if root.is_dir():
            source_files.update(p for p in root.rglob("*.json") if p.is_file())
            source_files.update(p for p in root.rglob("*.jsonl") if p.is_file())
    source_files.update([
        visual_production_root / "samples.jsonl", paths.audio / "canonical_clips.jsonl",
        paths.diarization / "inventory.json", paths.diarization / "raw_segments.jsonl",
        paths.diarization / "bound_segments.jsonl", paths.asr / "segments.jsonl",
        paths.root / "binding_audit_v1/segments.jsonl", paths.h3 / "samples.jsonl",
    ])
    hashes = {str(p): sha256_file(p) for p in sorted(source_files)}
    base = load_reconcile_sources(base_reconcile_root)
    override = load_reconcile_sources(override_reconcile_root) if override_reconcile_root else []
    effective = overlay_reconcile_sources(base, override)
    manifest = MimoCaseManifest(clip_uids=[r.clip_uid for r in base])
    stem_inventory, stems, _ = load_stem_shadow(shadow / "separation")
    provenance, _, _ = validate_stem_diarization_lineage(shadow / "diarization", expected_shadow_root=shadow)
    if Path(provenance.source_stem_root).resolve() != shadow / "separation":
        raise ValueError("prepared separation lineage differs")
    if [uid for uid in stem_inventory.clip_uids if uid in set(manifest.clip_uids)] != manifest.clip_uids:
        raise ValueError("prepared clips are not an ordered separation subset")
    base_inventory = build_mimo25_inventory(
        visual_production_root=visual_production_root, visual_runs_root=visual_runs_root,
        audio_production_root=audio_production_root, case_manifest=manifest,
    )
    if sha256_file(source_h3_root / "samples.jsonl") != base_inventory.source_h3_samples_sha256:
        raise ValueError("frozen H3 source differs from reconstruction source")
    jobs = build_stem_reconcile_jobs(
        base_inventory=usable_stem_reconcile_inventory(base_inventory, provenance),
        stem_diarization_root=shadow / "diarization", stem_asr_root=shadow / "asr", route=provenance.route,
    )
    samples = project_prepared_samples(jobs, [
        FinalH3SampleV2.model_validate_json(line)
        for line in (source_h3_root / "samples.jsonl").read_text().splitlines() if line.strip()
    ])
    values = base_inventory.model_dump(mode="json", exclude={"inventory_fingerprint"})
    if values["source_diarization_inventory_sha256"] is None:
        values.pop("source_diarization_inventory_sha256")
    values.update(jobs=[j.model_dump(mode="json") for j in jobs], clip_count=len(jobs))
    for name, path in (
        ("source_diarization_inventory_sha256", shadow / "diarization/inventory.json"),
        ("source_diarization_raw_segments_sha256", shadow / "diarization/raw_segments.jsonl"),
        ("source_diarization_bound_segments_sha256", shadow / "diarization/bound_segments.jsonl"),
        ("source_qwen3_asr_segments_sha256", shadow / "asr/segments.jsonl"),
    ):
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
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reuse-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        (stage / "h3").mkdir(parents=True)
        (stage / "h3/samples.jsonl").write_text(samples_text)
        (stage / "inventory.json").write_text(inventory.model_dump_json(indent=2) + "\n")
        (stage / "records.jsonl").write_text(records_text)
        (stage / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n")
        if any(sha256_file(Path(path)) != digest for path, digest in hashes.items()):
            raise ValueError("frozen preparation input changed")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return summary
