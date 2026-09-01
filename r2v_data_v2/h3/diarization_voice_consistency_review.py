from __future__ import annotations

import csv
import hashlib
import io
import json
import mimetypes
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path, PurePosixPath
from statistics import median
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import Field, model_validator

from r2v_data_v2.h3.diarization_voice_consistency_audit import (
    VoiceConsistencyAuditRecord,
    VoiceConsistencyAuditSummary,
)
from r2v_data_v2.h3.jea_final_renderer import FinalH3SampleV2
from r2v_data_v2.h3.schemas import SchemaModel

VOICE_CONSISTENCY_REVIEW_VERSION = (
    "r2v.h3.diarization_voice_consistency_human_review.1"
)
VOICE_CONSISTENCY_REVIEW_SUMMARY_VERSION = (
    "r2v.h3.diarization_voice_consistency_human_review_summary.1"
)
REVIEW_DECISIONS = ("same", "different", "uncertain")

ReviewDecision = Literal["same", "different", "uncertain"]


class VoiceConsistencyReviewAnnotation(SchemaModel):
    schema_version: Literal[
        "r2v.h3.diarization_voice_consistency_human_review.1"
    ] = VOICE_CONSISTENCY_REVIEW_VERSION
    clip_uid: str = Field(min_length=1)
    segment_id: str = Field(min_length=1)
    entity_id: str = Field(min_length=1)
    identity_scope: Literal["direct_anchor_present", "cluster_propagated_only"]
    cosine_similarity: float = Field(ge=-1, le=1, allow_inf_nan=False)
    source_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    segment_audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_voice_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: ReviewDecision
    notes: str = ""
    reviewed_at: datetime

    @model_validator(mode="after")
    def validate_annotation(self) -> VoiceConsistencyReviewAnnotation:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() != (
            UTC.utcoffset(self.reviewed_at)
        ):
            raise ValueError("voice-consistency review timestamp must be UTC")
        return self


class DecisionSimilaritySummary(SchemaModel):
    count: int = Field(ge=0)
    min: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    median: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    max: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_values(self) -> DecisionSimilaritySummary:
        values = (self.min, self.median, self.max)
        if self.count == 0 and any(value is not None for value in values):
            raise ValueError("empty review decision cannot publish similarity stats")
        if self.count > 0 and any(value is None for value in values):
            raise ValueError("review decision similarity statistics are incomplete")
        return self


class VoiceConsistencyReviewSummary(SchemaModel):
    schema_version: Literal[
        "r2v.h3.diarization_voice_consistency_human_review_summary.1"
    ] = VOICE_CONSISTENCY_REVIEW_SUMMARY_VERSION
    audit_root: str
    source_audio_production_root: str
    source_audit_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_audit_records_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    total: int = Field(ge=0)
    default_review_inventory_count: int = Field(ge=0)
    reviewed: int = Field(ge=0)
    unreviewed: int = Field(ge=0)
    decision_counts: dict[ReviewDecision, int]
    by_identity_scope: dict[str, dict[str, int]]
    by_duration_bucket: dict[str, dict[str, int]]
    decision_similarity: dict[ReviewDecision, DecisionSimilaritySummary]
    stale_annotation_count: int = Field(ge=0)
    similarity_threshold_applied: Literal[False] = False
    binding_modified: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> VoiceConsistencyReviewSummary:
        if self.reviewed + self.unreviewed != self.total:
            raise ValueError("voice-consistency review counts do not reconcile")
        if sum(self.decision_counts.values()) != self.reviewed:
            raise ValueError("voice-consistency decision counts do not reconcile")
        if self.default_review_inventory_count > self.total:
            raise ValueError("default review inventory exceeds audit records")
        return self


@dataclass(frozen=True)
class ReviewMediaAsset:
    path: Path
    sha256: str
    content_type: str


@dataclass(frozen=True)
class VoiceConsistencyReviewContext:
    audit_root: Path
    audit_summary: VoiceConsistencyAuditSummary
    records: tuple[VoiceConsistencyAuditRecord, ...]
    record_fingerprints: dict[tuple[str, str], str]
    target_video_by_clip: dict[str, Path]
    media_assets: dict[str, ReviewMediaAsset]
    inventory_payload: dict[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def record_fingerprint(record: VoiceConsistencyAuditRecord) -> str:
    return hashlib.sha256(
        _compact_json(record.model_dump(mode="json")).encode("utf-8")
    ).hexdigest()


def _read_jsonl(path: Path, model: type[SchemaModel]) -> list[SchemaModel]:
    with path.open(encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_audit_media_path(audit_root: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("audit segment audio path is unsafe")
    path = audit_root.joinpath(*relative.parts).resolve(strict=True)
    path.relative_to(audit_root)
    if not path.is_file():
        raise ValueError("audit segment audio is not a file")
    return path


def _media_url(
    *,
    fingerprint: str,
    role: Literal["target", "segment", "primary"],
    path: Path,
    sha256: str,
) -> str:
    suffix = path.suffix.lower()
    return f"/media/{fingerprint}/{role}{suffix}?v={sha256}"


def _target_video_map(
    *,
    h3_samples_path: Path,
) -> dict[str, Path]:
    samples = [
        FinalH3SampleV2.model_validate(item)
        for item in _read_jsonl(h3_samples_path, FinalH3SampleV2)
    ]
    output: dict[str, Path] = {}
    for sample in samples:
        target = Path(sample.target_video).expanduser().resolve(strict=True)
        if not target.is_file():
            raise ValueError("H3 target video is not a file")
        existing = output.setdefault(sample.clip_uid, target)
        if existing != target:
            raise ValueError("H3 samples disagree on target video for clip")
    return output


def _default_review_keys(
    records: tuple[VoiceConsistencyAuditRecord, ...],
) -> set[tuple[str, str]]:
    propagated = [
        item for item in records if item.identity_scope == "cluster_propagated_only"
    ]
    direct = sorted(
        (item for item in records if item.identity_scope == "direct_anchor_present"),
        key=lambda item: (item.cosine_similarity, item.clip_uid, item.segment_id),
    )[:10]
    return {(item.clip_uid, item.segment_id) for item in (*propagated, *direct)}


def load_review_context(audit_root: Path) -> VoiceConsistencyReviewContext:
    root = audit_root.expanduser().resolve(strict=True)
    summary_path = root / "summary.json"
    records_path = root / "records.jsonl"
    if not summary_path.is_file() or not records_path.is_file():
        raise ValueError("voice-consistency audit output is incomplete")
    audit_summary = VoiceConsistencyAuditSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    records = tuple(
        VoiceConsistencyAuditRecord.model_validate(item)
        for item in _read_jsonl(records_path, VoiceConsistencyAuditRecord)
    )
    if len(records) != audit_summary.audited_segment_count:
        raise ValueError("voice-consistency records differ from audit summary")
    keys = [(item.clip_uid, item.segment_id) for item in records]
    if len(keys) != len(set(keys)):
        raise ValueError("voice-consistency audit contains duplicate segments")
    production_root = Path(
        audit_summary.source_audio_production_root
    ).expanduser().resolve(strict=True)
    h3_samples_path = production_root / "h3/samples.jsonl"
    if not h3_samples_path.is_file():
        raise ValueError("source H3 samples are unavailable")
    if audit_summary.source_artifact_sha256.get("h3_samples") != _sha256_file(
        h3_samples_path
    ):
        raise ValueError("source H3 samples differ from audit provenance")
    target_video_by_clip = _target_video_map(h3_samples_path=h3_samples_path)
    missing_clips = {item.clip_uid for item in records} - set(target_video_by_clip)
    if missing_clips:
        raise ValueError("audit record has no matching H3 target video")
    default_keys = _default_review_keys(records)
    media_assets: dict[str, ReviewMediaAsset] = {}
    record_fingerprints: dict[tuple[str, str], str] = {}
    payload_items: list[dict[str, object]] = []
    for record in records:
        key = (record.clip_uid, record.segment_id)
        fingerprint = record_fingerprint(record)
        record_fingerprints[key] = fingerprint
        target_path = target_video_by_clip[record.clip_uid]
        target_sha = _sha256_file(target_path)
        segment_path = _safe_audit_media_path(root, record.segment_audio_path)
        if _sha256_file(segment_path) != record.segment_audio_sha256:
            raise ValueError("audit segment audio differs from record provenance")
        primary_path = Path(
            record.primary_voice_reference_path
        ).expanduser().resolve(strict=True)
        if not primary_path.is_file() or _sha256_file(primary_path) != (
            record.primary_voice_reference_sha256
        ):
            raise ValueError("primary voice differs from audit provenance")
        urls = {
            "target_video": _media_url(
                fingerprint=fingerprint,
                role="target",
                path=target_path,
                sha256=target_sha,
            ),
            "segment_audio": _media_url(
                fingerprint=fingerprint,
                role="segment",
                path=segment_path,
                sha256=record.segment_audio_sha256,
            ),
            "primary_voice": _media_url(
                fingerprint=fingerprint,
                role="primary",
                path=primary_path,
                sha256=record.primary_voice_reference_sha256,
            ),
        }
        for role, path, sha256 in (
            ("target", target_path, target_sha),
            ("segment", segment_path, record.segment_audio_sha256),
            ("primary", primary_path, record.primary_voice_reference_sha256),
        ):
            url = _media_url(
                fingerprint=fingerprint,
                role=role,
                path=path,
                sha256=sha256,
            )
            media_path = urlsplit(url).path
            content_type = mimetypes.guess_type(path.name)[0]
            if content_type is None:
                content_type = (
                    "video/mp4" if role == "target" else "audio/wav"
                )
            media_assets[media_path] = ReviewMediaAsset(
                path=path,
                sha256=sha256,
                content_type=content_type,
            )
        payload_items.append(
            {
                **record.model_dump(mode="json"),
                "record_fingerprint": fingerprint,
                "default_review_inventory": key in default_keys,
                "media": urls,
                "target_video_path": str(target_path),
            }
        )
    payload_items.sort(
        key=lambda item: (
            float(item["cosine_similarity"]),
            str(item["clip_uid"]),
            str(item["segment_id"]),
        )
    )
    return VoiceConsistencyReviewContext(
        audit_root=root,
        audit_summary=audit_summary,
        records=records,
        record_fingerprints=record_fingerprints,
        target_video_by_clip=target_video_by_clip,
        media_assets=media_assets,
        inventory_payload={
            "schema_version": "r2v.h3.diarization_voice_consistency_review_inventory.1",
            "items": payload_items,
            "default_review_inventory_count": len(default_keys),
            "similarity_threshold_applied": False,
            "binding_modified": False,
        },
    )


def _load_annotations(root: Path) -> list[VoiceConsistencyReviewAnnotation]:
    path = root / "human_review/annotations.jsonl"
    if not path.is_file():
        return []
    annotations = [
        VoiceConsistencyReviewAnnotation.model_validate(item)
        for item in _read_jsonl(path, VoiceConsistencyReviewAnnotation)
    ]
    keys = [
        (item.clip_uid, item.segment_id, item.record_fingerprint)
        for item in annotations
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("voice-consistency annotations contain duplicate records")
    return annotations


def _current_annotations(
    context: VoiceConsistencyReviewContext,
    annotations: list[VoiceConsistencyReviewAnnotation],
) -> dict[tuple[str, str], VoiceConsistencyReviewAnnotation]:
    output: dict[tuple[str, str], VoiceConsistencyReviewAnnotation] = {}
    records = {(item.clip_uid, item.segment_id): item for item in context.records}
    for annotation in annotations:
        key = (annotation.clip_uid, annotation.segment_id)
        record = records.get(key)
        if record is None:
            continue
        if (
            context.record_fingerprints[key] == annotation.record_fingerprint
            and record.source_audio_sha256 == annotation.source_audio_sha256
            and record.segment_audio_sha256 == annotation.segment_audio_sha256
            and record.primary_voice_reference_sha256
            == annotation.primary_voice_reference_sha256
        ):
            output[key] = annotation
    return output


def _decision_similarity(
    annotations: list[VoiceConsistencyReviewAnnotation],
) -> DecisionSimilaritySummary:
    values = sorted(item.cosine_similarity for item in annotations)
    if not values:
        return DecisionSimilaritySummary(count=0)
    return DecisionSimilaritySummary(
        count=len(values),
        min=values[0],
        median=float(median(values)),
        max=values[-1],
    )


def _group_counts(
    records: tuple[VoiceConsistencyAuditRecord, ...],
    current: dict[tuple[str, str], VoiceConsistencyReviewAnnotation],
    *,
    field: Literal["identity_scope", "duration_bucket"],
) -> dict[str, dict[str, int]]:
    groups = sorted({str(getattr(item, field)) for item in records})
    output: dict[str, dict[str, int]] = {}
    for group in groups:
        members = [item for item in records if str(getattr(item, field)) == group]
        annotations = [
            current[(item.clip_uid, item.segment_id)]
            for item in members
            if (item.clip_uid, item.segment_id) in current
        ]
        output[group] = {
            "total": len(members),
            "reviewed": len(annotations),
            "unreviewed": len(members) - len(annotations),
            **{
                decision: sum(item.decision == decision for item in annotations)
                for decision in REVIEW_DECISIONS
            },
        }
    return output


def build_review_summary(
    context: VoiceConsistencyReviewContext,
    annotations: list[VoiceConsistencyReviewAnnotation],
) -> VoiceConsistencyReviewSummary:
    current = _current_annotations(context, annotations)
    current_values = list(current.values())
    decision_counts = {
        decision: sum(item.decision == decision for item in current_values)
        for decision in REVIEW_DECISIONS
    }
    default_count = int(
        context.inventory_payload["default_review_inventory_count"]
    )
    return VoiceConsistencyReviewSummary(
        audit_root=str(context.audit_root),
        source_audio_production_root=(
            context.audit_summary.source_audio_production_root
        ),
        source_audit_summary_sha256=_sha256_file(context.audit_root / "summary.json"),
        source_audit_records_sha256=_sha256_file(context.audit_root / "records.jsonl"),
        total=len(context.records),
        default_review_inventory_count=default_count,
        reviewed=len(current),
        unreviewed=len(context.records) - len(current),
        decision_counts=decision_counts,
        by_identity_scope=_group_counts(
            context.records,
            current,
            field="identity_scope",
        ),
        by_duration_bucket=_group_counts(
            context.records,
            current,
            field="duration_bucket",
        ),
        decision_similarity={
            decision: _decision_similarity(
                [item for item in current_values if item.decision == decision]
            )
            for decision in REVIEW_DECISIONS
        },
        stale_annotation_count=len(annotations) - len(current),
    )


def _annotations_csv(annotations: list[VoiceConsistencyReviewAnnotation]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    fields = (
        "schema_version",
        "clip_uid",
        "segment_id",
        "entity_id",
        "identity_scope",
        "cosine_similarity",
        "source_audio_sha256",
        "segment_audio_sha256",
        "primary_voice_reference_sha256",
        "record_fingerprint",
        "decision",
        "notes",
        "reviewed_at",
    )
    writer.writerow(fields)
    for item in annotations:
        value = item.model_dump(mode="json")
        writer.writerow(value[field] for field in fields)
    return output.getvalue()


def _write_review_outputs(
    context: VoiceConsistencyReviewContext,
    annotations: list[VoiceConsistencyReviewAnnotation],
) -> VoiceConsistencyReviewSummary:
    ordered = sorted(
        annotations,
        key=lambda item: (item.clip_uid, item.segment_id, item.record_fingerprint),
    )
    summary = build_review_summary(context, ordered)
    review_root = context.audit_root / "human_review"
    _atomic_write_text(
        review_root / "annotations.jsonl",
        "".join(
            _compact_json(item.model_dump(mode="json")) + "\n" for item in ordered
        ),
    )
    _atomic_write_text(review_root / "annotations.csv", _annotations_csv(ordered))
    _atomic_write_text(
        review_root / "summary.json",
        summary.model_dump_json(indent=2) + "\n",
    )
    return summary


def initialize_review(
    audit_root: Path,
) -> tuple[VoiceConsistencyReviewContext, VoiceConsistencyReviewSummary]:
    context = load_review_context(audit_root)
    annotations = _load_annotations(context.audit_root)
    return context, _write_review_outputs(context, annotations)


def current_reviews(context: VoiceConsistencyReviewContext) -> dict[str, object]:
    annotations = _load_annotations(context.audit_root)
    current = _current_annotations(context, annotations)
    return {
        "annotations": [
            current[key].model_dump(mode="json")
            for key in sorted(current)
        ],
        "stale_annotation_count": len(annotations) - len(current),
    }


def save_review(
    context: VoiceConsistencyReviewContext,
    payload: dict[str, object],
    *,
    reviewed_at: datetime | None = None,
) -> tuple[VoiceConsistencyReviewAnnotation, VoiceConsistencyReviewSummary]:
    clip_uid = payload.get("clip_uid")
    segment_id = payload.get("segment_id")
    if not isinstance(clip_uid, str) or not isinstance(segment_id, str):
        raise TypeError("voice-consistency review identity is incomplete")
    key = (clip_uid, segment_id)
    records = {(item.clip_uid, item.segment_id): item for item in context.records}
    record = records.get(key)
    if record is None:
        raise ValueError("voice-consistency review references an unknown segment")
    fingerprint = context.record_fingerprints[key]
    for field, expected in (
        ("record_fingerprint", fingerprint),
        ("source_audio_sha256", record.source_audio_sha256),
        ("segment_audio_sha256", record.segment_audio_sha256),
        (
            "primary_voice_reference_sha256",
            record.primary_voice_reference_sha256,
        ),
    ):
        if payload.get(field) != expected:
            raise ValueError("voice-consistency review fingerprint is stale")
    annotation = VoiceConsistencyReviewAnnotation.model_validate(
        {
            "clip_uid": record.clip_uid,
            "segment_id": record.segment_id,
            "entity_id": record.entity_id,
            "identity_scope": record.identity_scope,
            "cosine_similarity": record.cosine_similarity,
            "source_audio_sha256": record.source_audio_sha256,
            "segment_audio_sha256": record.segment_audio_sha256,
            "primary_voice_reference_sha256": (
                record.primary_voice_reference_sha256
            ),
            "record_fingerprint": fingerprint,
            "decision": payload.get("decision"),
            "notes": payload.get("notes", ""),
            "reviewed_at": reviewed_at or datetime.now(UTC),
        }
    )
    existing = _load_annotations(context.audit_root)
    by_key = {
        (item.clip_uid, item.segment_id, item.record_fingerprint): item
        for item in existing
    }
    by_key[(clip_uid, segment_id, fingerprint)] = annotation
    summary = _write_review_outputs(context, list(by_key.values()))
    return annotation, summary


def render_review_html() -> str:
    decisions = "".join(
        f"<button type='button' class='decision {decision}' data-decision='{decision}'>"
        f"{decision.upper()}</button>"
        for decision in REVIEW_DECISIONS
    )
    script = r"""
const state = {items: [], annotations: new Map(), visible: [], index: 0, pending: null};
const byId = id => document.getElementById(id);
const keyFor = item => `${item.clip_uid}\u0000${item.segment_id}`;
function applyFilter() {
  const mode = byId('scope-filter').value;
  const descending = byId('sort-order').value === 'desc';
  state.visible = state.items.filter(item => {
    if (mode === 'default') return item.default_review_inventory;
    if (mode === 'propagated') return item.identity_scope === 'cluster_propagated_only';
    if (mode === 'direct') return item.identity_scope === 'direct_anchor_present';
    return true;
  }).sort((a, b) => {
    const delta = a.cosine_similarity - b.cosine_similarity;
    const stable = delta || a.clip_uid.localeCompare(b.clip_uid) || a.segment_id.localeCompare(b.segment_id);
    return descending ? -stable : stable;
  });
  state.index = Math.min(state.index, Math.max(0, state.visible.length - 1));
  render();
}
function selectDecision(decision) {
  state.pending = decision;
  document.querySelectorAll('.decision').forEach(node => node.classList.toggle('selected', node.dataset.decision === decision));
}
function render() {
  const item = state.visible[state.index];
  byId('position').textContent = state.visible.length ? `${state.index + 1} / ${state.visible.length}` : '0 / 0';
  byId('case').hidden = !item;
  if (!item) return;
  byId('identity').textContent = `${item.clip_uid} / ${item.segment_id}`;
  byId('entity').textContent = item.entity_id;
  byId('cluster').textContent = item.speaker_cluster_id;
  byId('scope').textContent = item.identity_scope;
  byId('duration').textContent = item.duration_seconds.toFixed(6);
  byId('anchor').textContent = item.direct_anchor_seconds.toFixed(6);
  byId('similarity').textContent = item.cosine_similarity.toFixed(6);
  byId('target-video').src = item.media.target_video;
  byId('segment-audio').src = item.media.segment_audio;
  byId('primary-audio').src = item.media.primary_voice;
  const annotation = state.annotations.get(keyFor(item));
  state.pending = annotation ? annotation.decision : null;
  byId('notes').value = annotation ? annotation.notes : '';
  byId('saved-at').textContent = annotation ? `Saved at ${annotation.reviewed_at}` : 'Unreviewed';
  selectDecision(state.pending);
}
function move(delta) {
  if (!state.visible.length) return;
  state.index = Math.max(0, Math.min(state.visible.length - 1, state.index + delta));
  render();
}
async function save(andNext) {
  const item = state.visible[state.index];
  if (!item || !state.pending) { alert('Choose SAME, DIFFERENT, or UNCERTAIN.'); return; }
  const payload = {
    clip_uid: item.clip_uid,
    segment_id: item.segment_id,
    record_fingerprint: item.record_fingerprint,
    source_audio_sha256: item.source_audio_sha256,
    segment_audio_sha256: item.segment_audio_sha256,
    primary_voice_reference_sha256: item.primary_voice_reference_sha256,
    decision: state.pending,
    notes: byId('notes').value
  };
  const response = await fetch('/api/review', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const result = await response.json();
  if (!response.ok) { alert(result.error || 'Save failed.'); return; }
  state.annotations.set(keyFor(item), result.annotation);
  updateDashboard(result.summary);
  if (andNext) move(1); else render();
}
function updateDashboard(summary) {
  byId('dashboard').textContent = `Reviewed ${summary.reviewed} / ${summary.total} · SAME ${summary.decision_counts.same} · DIFFERENT ${summary.decision_counts.different} · UNCERTAIN ${summary.decision_counts.uncertain}`;
}
async function load() {
  const [inventoryResponse, reviewsResponse, summaryResponse] = await Promise.all([
    fetch('/api/inventory', {cache: 'no-store'}),
    fetch('/api/reviews', {cache: 'no-store'}),
    fetch('/api/summary', {cache: 'no-store'})
  ]);
  if (!inventoryResponse.ok || !reviewsResponse.ok || !summaryResponse.ok) throw new Error('Review API unavailable');
  const inventory = await inventoryResponse.json();
  const reviews = await reviewsResponse.json();
  const summary = await summaryResponse.json();
  state.items = inventory.items;
  reviews.annotations.forEach(item => state.annotations.set(`${item.clip_uid}\u0000${item.segment_id}`, item));
  updateDashboard(summary);
  applyFilter();
}
document.querySelectorAll('.decision').forEach(node => node.addEventListener('click', () => selectDecision(node.dataset.decision)));
byId('previous').addEventListener('click', () => move(-1));
byId('next').addEventListener('click', () => move(1));
byId('save').addEventListener('click', () => save(false));
byId('save-next').addEventListener('click', () => save(true));
byId('scope-filter').addEventListener('change', () => {state.index = 0; applyFilter();});
byId('sort-order').addEventListener('change', () => {state.index = 0; applyFilter();});
document.addEventListener('keydown', event => {
  if (event.key === '1') selectDecision('same');
  else if (event.key === '2') selectDecision('different');
  else if (event.key === '3') selectDecision('uncertain');
  else if (event.key === 'Enter' && !event.shiftKey) {event.preventDefault(); save(true);}
});
load().catch(error => {byId('dashboard').textContent = error.message;});
"""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>H3 diarization voice consistency review</title><style>"
        "body{font:15px system-ui;margin:0;background:#f2f3f5;color:#171717}"
        "header{position:sticky;top:0;z-index:4;background:#fff;border-bottom:1px solid #bbb;padding:12px 18px}"
        ".toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}"
        "main{padding:18px}.layout{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(320px,.75fr);gap:18px}"
        ".panel{background:#fff;border:1px solid #bbb;padding:16px}video{width:100%;max-height:62vh;background:#000}"
        "audio{width:100%;margin:8px 0 22px}.similarity{font-size:44px;font-weight:750}"
        ".facts{display:grid;grid-template-columns:max-content 1fr;gap:6px 12px;margin:14px 0}"
        ".buttons{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}"
        ".decision{font-size:20px;font-weight:700;padding:18px;border:2px solid #555;background:#fff}"
        ".decision.selected{background:#161616;color:#fff}.same.selected{background:#176b31}.different.selected{background:#a12622}.uncertain.selected{background:#765b00}"
        "textarea{width:100%;min-height:100px;box-sizing:border-box}.nav{display:flex;gap:10px;align-items:center;margin-top:12px}"
        ".nav button{padding:10px 16px}.primary{background:#111;color:#fff;border:1px solid #111}"
        "@media(max-width:850px){.layout{grid-template-columns:1fr}.buttons{grid-template-columns:1fr}}"
        "</style></head><body><header><h1>Speaker propagation voice-consistency review</h1>"
        "<div id='dashboard'>Loading…</div><div class='toolbar'>"
        "<label>Inventory <select id='scope-filter'><option value='default'>Default review inventory</option>"
        "<option value='propagated'>Propagated only</option><option value='direct'>Direct anchor</option><option value='all'>All</option></select></label>"
        "<label>Sort <select id='sort-order'><option value='asc'>Similarity ascending</option><option value='desc'>Similarity descending</option></select></label>"
        "<strong id='position'>0 / 0</strong></div></header><main><section id='case' class='layout' hidden>"
        "<div class='panel'><h2 id='identity'></h2><video id='target-video' controls preload='metadata'></video></div>"
        "<div class='panel'><div class='similarity' id='similarity'></div><div class='facts'>"
        "<b>Entity</b><span id='entity'></span><b>Speaker cluster</b><span id='cluster'></span>"
        "<b>Identity scope</b><span id='scope'></span><b>Duration seconds</b><span id='duration'></span>"
        "<b>Direct anchor seconds</b><span id='anchor'></span></div>"
        "<h3>Segment audio</h3><audio id='segment-audio' controls preload='metadata'></audio>"
        "<h3>Primary voice reference</h3><audio id='primary-audio' controls preload='metadata'></audio>"
        "<div class='buttons'>" + decisions + "</div><label>Notes<textarea id='notes'></textarea></label>"
        "<p id='saved-at'>Unreviewed</p><div class='nav'><button id='previous'>Previous</button>"
        "<button id='next'>Next</button><button id='save'>Save</button>"
        "<button id='save-next' class='primary'>Save &amp; Next</button></div>"
        "<p>Keyboard: 1 SAME · 2 DIFFERENT · 3 UNCERTAIN · Enter Save &amp; Next</p>"
        "</div></section></main><script>" + script + "</script></body></html>"
    )


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: HTTPStatus,
    value: object,
) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(payload)


def _range_bounds(value: str, size: int) -> tuple[int, int] | None:
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if match is None or size <= 0:
        return None
    start_text, end_text = match.groups()
    if not start_text and not end_text:
        return None
    if start_text:
        start = int(start_text)
        end = size - 1 if not end_text else min(int(end_text), size - 1)
    else:
        length = int(end_text)
        if length <= 0:
            return None
        start = max(0, size - length)
        end = size - 1
    if start < 0 or start >= size or end < start:
        return None
    return start, end


def make_review_handler(
    context: VoiceConsistencyReviewContext,
) -> type[BaseHTTPRequestHandler]:
    review_html = render_review_html().encode("utf-8")

    class ReviewHandler(BaseHTTPRequestHandler):
        def _send_bytes(
            self,
            payload: bytes,
            *,
            content_type: str,
            include_body: bool,
        ) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if include_body:
                self.wfile.write(payload)

        def _send_media(
            self,
            asset: ReviewMediaAsset,
            *,
            include_body: bool,
        ) -> None:
            size = asset.path.stat().st_size
            range_header = self.headers.get("Range")
            bounds = None if range_header is None else _range_bounds(range_header, size)
            if range_header is not None and bounds is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return
            start, end = (0, size - 1) if bounds is None else bounds
            length = end - start + 1
            self.send_response(
                HTTPStatus.OK if bounds is None else HTTPStatus.PARTIAL_CONTENT
            )
            self.send_header("Content-Type", asset.content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            if bounds is not None:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if not include_body:
                return
            with asset.path.open("rb") as source:
                source.seek(start)
                remaining = length
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise OSError("allowlisted media ended before declared size")
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _get(self, *, include_body: bool) -> None:
            request_path = unquote(urlsplit(self.path).path)
            parts = PurePosixPath(request_path.removeprefix("/")).parts
            if any(part in {"", ".", ".."} for part in parts):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if request_path in {"/", "/review.html"}:
                self._send_bytes(
                    review_html,
                    content_type="text/html; charset=utf-8",
                    include_body=include_body,
                )
                return
            if request_path == "/api/inventory":
                _json_response(self, HTTPStatus.OK, context.inventory_payload)
                return
            if request_path == "/api/reviews":
                _json_response(self, HTTPStatus.OK, current_reviews(context))
                return
            if request_path == "/api/summary":
                summary = build_review_summary(
                    context,
                    _load_annotations(context.audit_root),
                )
                _json_response(
                    self,
                    HTTPStatus.OK,
                    summary.model_dump(mode="json"),
                )
                return
            asset = context.media_assets.get(request_path)
            if asset is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self._send_media(asset, include_body=include_body)

        def do_GET(self) -> None:
            self._get(include_body=True)

        def do_HEAD(self) -> None:
            self._get(include_body=False)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024 * 1024:
                    raise ValueError("invalid review request length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise TypeError("review request must be a JSON object")
                annotation, summary = save_review(context, payload)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
                _json_response(
                    self,
                    HTTPStatus.BAD_REQUEST,
                    {"error": f"{type(exc).__name__}: {exc}"},
                )
                return
            _json_response(
                self,
                HTTPStatus.OK,
                {
                    "annotation": annotation.model_dump(mode="json"),
                    "summary": summary.model_dump(mode="json"),
                },
            )

    return ReviewHandler


__all__ = [
    "DecisionSimilaritySummary",
    "ReviewMediaAsset",
    "VoiceConsistencyReviewAnnotation",
    "VoiceConsistencyReviewContext",
    "VoiceConsistencyReviewSummary",
    "build_review_summary",
    "current_reviews",
    "initialize_review",
    "load_review_context",
    "make_review_handler",
    "record_fingerprint",
    "render_review_html",
    "save_review",
]
