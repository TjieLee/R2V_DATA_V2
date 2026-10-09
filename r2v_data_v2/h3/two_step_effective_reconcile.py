"""Read-only cached Two-step replay and frozen-record overlay publication."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from r2v_data_v2.h3.audio_reuse_prepared import _hash
from r2v_data_v2.h3.mimo25_av_reconcile import MimoClipJob
from r2v_data_v2.h3.mimo25_backend import (
    _DIALOGUE,
    _REVIEW_ONLY_ISSUES,
    MimoBackendConfig,
    MimoBackendFailure,
    direct_speech_facts,
    protect_direct_dialogue,
    validate_annotation,
)
from r2v_data_v2.h3.mimo25_stem_shadow import (
    MimoStemReconcileRecord,
    MimoStemReconcileSummary,
    _publish_directory,
    _write_json,
    _write_jsonl,
)
from r2v_data_v2.h3.mimo26_two_step_backend import (
    TWO_STEP_BACKEND_VERSION,
    TWO_STEP_PROMPT_VERSION,
    TwoStepOpenAIMimo26Backend,
)
from r2v_data_v2.h3.resolved_audio_stems import load_stem_source
from r2v_data_v2.h3.speaker_ownership import (
    two_step_identity_restricted_groups,
    unconfirmed_visible_binding_segments,
)

DEFAULT_REPLAY_CLIP_UIDS = (
    "aed328a5a4eb2ff45d0d4d66", "02750d804022e9e8ba5f2964",
    "4e0506dfb9454536cc3c37b1", "009a0523c7a1332b52fcc82a",
)


class _ReplayMedia:
    def __init__(self, provenance):
        self.mode = provenance.media_mode
        self.media_root = Path(provenance.media_root)
        self.media_base_url = provenance.media_base_url

    def resolve(self, path):
        return "http://cached.invalid/" + Path(path).name


def replay_two_step_record(job: MimoClipJob, record: MimoStemReconcileRecord) -> MimoStemReconcileRecord:
    """Reuse two saved completions; retain their original HTTP/token audit."""
    if record.clip_uid != job.clip_uid or record.source_job_fingerprint != job.request_fingerprint:
        raise ValueError("cached replay source job fingerprint mismatch")
    if record.visual_raw_response is None or record.speech_av_raw_response is None:
        raise ValueError(f"{job.clip_uid}: complete Visual and Joint raw required for CPU replay")
    provenance = record.backend_provenance
    cached = iter(zip(record.raw_responses, record.diagnostics, strict=True))

    def create(**request):
        raw, diagnostic = next(cached)
        usage = diagnostic.usage.model_dump()
        return {
            "choices": [{"message": {"content": raw}, "finish_reason": diagnostic.finish_reason}],
            "usage": {**usage, "prompt_tokens_details": {key: usage[key] for key in
                      ("image_tokens", "video_tokens", "audio_tokens", "cached_tokens")},
                      "completion_tokens_details": {"reasoning_tokens": usage["reasoning_tokens"]}},
        }

    config = MimoBackendConfig(
        api_key="cached-cpu-replay", media_resolver=_ReplayMedia(provenance),
        transport=provenance.transport, base_url=provenance.base_url, model=provenance.model,
        video_fps=provenance.video_fps, media_resolution=provenance.media_resolution,
        temperature=provenance.temperature, max_completion_tokens=provenance.max_completion_tokens,
        thinking=provenance.thinking.type, icl="official_ref2va_v1" if provenance.icl_version else "none",
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    backend = TwoStepOpenAIMimo26Backend(config, stem_records_by_clip={}, client=client)
    labels, entities = _labels_and_entities(job)
    failure = None
    try:
        result = backend.reconcile(
            job, segment_ids=[s.segment_id for s in job.segments],
            transcribed_segment_ids=[s.segment_id for s in job.segments if s.asr_status == "transcribed"],
            allowed_entity_ids=entities, allowed_reference_labels=labels,
            auxiliary_audio_paths={kind: Path(kind + ".flac") for kind in ("speech", "music", "sfx")},
        )
    except MimoBackendFailure as exc:
        result, failure = exc, exc
    if list(result.raw_responses) != record.raw_responses or len(result.diagnostics) != len(record.diagnostics):
        raise ValueError(f"{job.clip_uid}: cached replay raw/turn inventory differs")
    values = record.model_dump(mode="json", exclude={"record_fingerprint"})
    values.update(
        backend_provenance=backend.provenance.model_dump(mode="json"),
        status="failed" if failure else "ready",
        annotation=result.annotation.model_dump(mode="json") if result.annotation is not None else None,
        failure_code=failure.code if failure else None, failure_reason=failure.reason if failure else None,
        failure_issues=[issue.to_dict() for issue in failure.issues] if failure else [],
    )
    # The replay's synthetic completion objects are not a new inference audit.
    for saved, replayed in zip(values["diagnostics"], result.diagnostics, strict=True):
        saved["warnings"] = list(dict.fromkeys([*saved["warnings"], *replayed.warnings]))
    return MimoStemReconcileRecord(**values, record_fingerprint=_hash(values))


def _labels_and_entities(job):
    labels = {r.picture_label for r in job.reference_images} | {s.subject_label for s in job.reference_subjects}
    contract = TwoStepOpenAIMimo26Backend.build_compact_task_contract(job)
    return labels, set(contract["allowed_speaker_bindable_entity_ids"])


def _validate_ready_record(job, record):
    if record.status != "ready":
        return
    annotation = record.annotation
    labels, entities = _labels_and_entities(job)
    transcribed = [s for s in job.segments if s.asr_status == "transcribed"]
    issues = validate_annotation(
        annotation, segment_ids=[s.segment_id for s in job.segments],
        segment_intervals={s.segment_id: (s.start_time, s.end_time) for s in job.segments},
        transcribed_segment_ids=[s.segment_id for s in transcribed],
        authoritative_transcripts=[s.asr_text for s in transcribed],
        allowed_entity_ids=entities, allowed_reference_labels=labels, reference_subjects=job.reference_subjects,
        target_duration_seconds=job.target_duration_seconds, binding_evidence_mode=job.binding_evidence_mode,
    )
    review_only = _REVIEW_ONLY_ISSUES
    if not transcribed:
        review_only = review_only | {"segment_inventory_mismatch", "visual_segment_inventory_mismatch"}
    unconfirmed = unconfirmed_visible_binding_segments(annotation)
    issues = [i for i in issues if i.code not in review_only and not (
        i.code == "visible_entity_binding_not_permitted" and i.field in unconfirmed
    )]
    _, direct_issues, _ = protect_direct_dialogue(
        annotation.h3_semantics.shot1_caption, direct_speech_facts(annotation, list(job.segments)),
        allowed_labels=labels,
    )
    issues.extend(direct_issues)
    expected = [f"<d>[{s.asr_language or 'Unknown'}] {s.asr_text}</d>" for s in transcribed]
    actual = [m.group() for m in _DIALOGUE.finditer(annotation.h3_semantics.shot1_caption)]
    if issues or expected != actual:
        raise ValueError(f"{job.clip_uid}: ready annotation validation failed: "
                         f"{[i.to_dict() for i in issues]}; exact_dialogue_matches={expected == actual}")


def _load_root(root):
    records = [MimoStemReconcileRecord.model_validate_json(line)
               for line in (root / "records.jsonl").read_text().splitlines() if line.strip()]
    summary = MimoStemReconcileSummary.model_validate_json((root / "summary.json").read_text())
    ids = [r.clip_uid for r in records]
    counts = Counter(r.status for r in records)
    if len(ids) != len(set(ids)) or summary.processed_clip_uids != ids or (
        summary.ready_count, summary.failed_count
    ) != (counts["ready"], counts["failed"]):
        raise ValueError(f"{root}: summary and records differ")
    for name in ("model_call_count", "visual_model_call_count", "av_model_call_count",
                 "audio_model_call_count", "text_model_call_count"):
        if getattr(summary, name) != sum(getattr(r, name) for r in records):
            raise ValueError(f"{root}: {name} summary differs")
    if any(r.backend_provenance.prompt_version != TWO_STEP_PROMPT_VERSION for r in records):
        raise ValueError(f"{root}: expected Two-step Joint v2 provenance")
    contract = json.loads((root / "source_contract.json").read_text())
    jobs = [MimoClipJob.model_validate(row) for row in contract["jobs"]]
    if (contract.get("schema_version") != "r2v.h3.mimo25_stem_source_contract.1"
            or contract.get("binding_evidence_mode") != "none" or any(j.binding_evidence_mode != "none" for j in jobs)
            or contract["clip_uids"] != ids or [j.clip_uid for j in jobs] != ids):
        raise ValueError(f"{root}: frozen source order differs")
    if any(j.request_fingerprint != r.source_job_fingerprint for j, r in zip(jobs, records, strict=True)):
        raise ValueError(f"{root}: source job fingerprint mismatch")
    return jobs, records, summary


def build_effective_reconcile(
    *, base_root: Path, override_root: Path, output_root: Path,
    replay_clip_uids=DEFAULT_REPLAY_CLIP_UIDS,
) -> MimoStemReconcileSummary:
    base, override, output = (p.expanduser().resolve() for p in (base_root, override_root, output_root))
    if output.exists():
        raise FileExistsError(output)
    jobs, base_records, original_summary = _load_root(base)
    _, overrides, _ = _load_root(override)
    by_clip = {r.clip_uid: r for r in base_records}
    replacements = {r.clip_uid: r for r in overrides}
    replay_ids = set(replay_clip_uids)
    if not replay_ids.union(replacements).issubset(by_clip) or replay_ids.intersection(replacements):
        raise ValueError("unknown or overlapping replay/override clips")
    _, stems, _ = load_stem_source(base.parent / "resolved_stems_v1")
    stem_by_clip = {s.clip_uid: s for s in stems}
    for row in [*base_records, *overrides]:
        stem = stem_by_clip.get(row.clip_uid)
        if stem is None or row.source_stem_record_fingerprint != stem.record_fingerprint:
            raise ValueError(f"{row.clip_uid}: source stem fingerprint mismatch")
        if row.source_job_fingerprint != by_clip[row.clip_uid].source_job_fingerprint:
            raise ValueError(f"{row.clip_uid}: override source job fingerprint mismatch")
    effective, audit = [], []
    for job, original in zip(jobs, base_records, strict=True):
        source = replacements.get(job.clip_uid, original)
        root = override if job.clip_uid in replacements else base
        operation = "real_override" if root == override else "original"
        row = source
        if job.clip_uid in replay_ids:
            row = replay_two_step_record(job, source)
            operation = "cpu_replay"
        _validate_ready_record(job, row)
        corrections = {}
        for diagnostic in row.diagnostics:
            for warning in diagnostic.warnings:
                if warning.startswith("deterministic_correction_count:"):
                    key, count = warning.removeprefix("deterministic_correction_count:").rsplit("=", 1)
                    corrections[key] = int(count)
        audit.append({
            "clip_uid": job.clip_uid, "operation": operation, "source_records_path": str(root / "records.jsonl"),
            "source_record_fingerprint": source.record_fingerprint,
            "source_backend_provenance": source.backend_provenance.model_dump(mode="json"),
            "postprocess_backend_version": TWO_STEP_BACKEND_VERSION if operation == "cpu_replay" else None,
            "effective_validation_backend_version": TWO_STEP_BACKEND_VERSION,
            "effective_record_fingerprint": row.record_fingerprint, "status": row.status,
            "failure_reason": row.failure_reason, "correction_counts": corrections,
            "identity_publication_restricted_groups": sorted(two_step_identity_restricted_groups(
                row.annotation, row.backend_provenance,
            )) if row.annotation is not None else [],
            "new_model_call_count": 0,
        })
        effective.append(row)
    values = original_summary.model_dump(mode="json")
    counts = Counter(r.status for r in effective)
    values.update(ready_count=counts["ready"], failed_count=counts["failed"])
    for name in ("model_call_count", "visual_model_call_count", "av_model_call_count",
                 "audio_model_call_count", "text_model_call_count"):
        values[name] = sum(getattr(r, name) for r in effective)
    summary = MimoStemReconcileSummary.model_validate(values)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        _write_jsonl(stage / "records.jsonl", effective)
        _write_json(stage / "summary.json", summary)
        (stage / "source_contract.json").write_bytes((base / "source_contract.json").read_bytes())
        _write_json(stage / "effective_sources.json", {
            "new_model_call_count": 0, "retained_inference_model_call_count": summary.model_call_count,
            "records": audit,
        })
        _load_root(stage)
        _publish_directory(stage, output, overwrite=False)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return summary
