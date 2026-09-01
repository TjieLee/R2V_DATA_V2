from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import mimetypes
import os
import shutil
import uuid
from collections import Counter
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.parse import unquote, urlsplit

from pydantic import Field, model_validator

from r2v_data_v2.h3.qwen38_h3_recaption import (
    QWEN38_RECAPTION_MATERIALIZER_VERSION,
    QWEN38_RECAPTION_POLICY_VERSION,
    QWEN38_RECAPTION_PROMPT_VERSION,
    Qwen38RecaptionRecord,
    Qwen38RecaptionSummary,
)
from r2v_data_v2.h3.schemas import SchemaModel

HUMAN_REVIEW_VERSION = "r2v.h3.qwen38_human_review.1"
HUMAN_REVIEW_SUMMARY_VERSION = "r2v.h3.qwen38_human_review_summary.1"
REVIEW_DECISIONS = ("pass", "issue", "skip")
REVIEW_SEVERITIES = ("minor", "major", "blocker")
REVIEW_ISSUE_TAGS = (
    "video_reference_mismatch",
    "wrong_target_video",
    "wrong_reference_image",
    "visual_caption_issue",
    "visual_hallucination",
    "reference_contract_issue",
    "audio_semantics_issue",
    "audio_event_oversegmentation",
    "audio_hallucination",
    "music_issue",
    "speaker_binding_issue",
    "asr_issue",
    "dialogue_timing_issue",
    "h3_format_issue",
    "model_failure",
    "other",
)

ReviewDecision = Literal["pass", "issue", "skip"]
ReviewSeverity = Literal["minor", "major", "blocker"]
ReviewIssueTag = Literal[
    "video_reference_mismatch",
    "wrong_target_video",
    "wrong_reference_image",
    "visual_caption_issue",
    "visual_hallucination",
    "reference_contract_issue",
    "audio_semantics_issue",
    "audio_event_oversegmentation",
    "audio_hallucination",
    "music_issue",
    "speaker_binding_issue",
    "asr_issue",
    "dialogue_timing_issue",
    "h3_format_issue",
    "model_failure",
    "other",
]


class Qwen38HumanReviewAnnotation(SchemaModel):
    schema_version: Literal["r2v.h3.qwen38_human_review.1"] = HUMAN_REVIEW_VERSION
    sample_id: str = Field(min_length=1)
    clip_uid: str = Field(min_length=1)
    request_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: ReviewDecision
    severity: ReviewSeverity | None = None
    issue_tags: list[ReviewIssueTag] = Field(default_factory=list)
    notes: str = ""
    reviewed_at: datetime

    @model_validator(mode="after")
    def validate_annotation(self) -> Qwen38HumanReviewAnnotation:
        if self.reviewed_at.tzinfo is None:
            raise ValueError("human review timestamp must include UTC timezone")
        if self.reviewed_at.utcoffset() != UTC.utcoffset(self.reviewed_at):
            raise ValueError("human review timestamp must be UTC")
        if len(self.issue_tags) != len(set(self.issue_tags)):
            raise ValueError("human review issue tags must be unique")
        if self.decision == "issue":
            if self.severity is None or not self.issue_tags:
                raise ValueError("issue review requires severity and issue tags")
        elif self.severity is not None or self.issue_tags:
            raise ValueError("pass/skip review cannot publish issue severity or tags")
        return self


class Qwen38HumanReviewSummary(SchemaModel):
    schema_version: Literal["r2v.h3.qwen38_human_review_summary.1"] = (
        HUMAN_REVIEW_SUMMARY_VERSION
    )
    batch: dict[str, object]
    scope: dict[str, object]
    model: dict[str, object]
    human_review: dict[str, object]
    generated_at: datetime


def _compact_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[object]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def _load_output(
    output_root: Path,
) -> tuple[list[Qwen38RecaptionRecord], Qwen38RecaptionSummary]:
    root = output_root.expanduser().resolve(strict=True)
    records_path = root / "records.jsonl"
    summary_path = root / "summary.json"
    if not records_path.is_file() or not summary_path.is_file():
        raise ValueError("Qwen3.8 recaption review output is incomplete")
    records = [
        Qwen38RecaptionRecord.model_validate(value)
        for value in _read_jsonl(records_path)
    ]
    summary = Qwen38RecaptionSummary.model_validate_json(
        summary_path.read_text(encoding="utf-8")
    )
    if summary.case_count != len(records):
        raise ValueError("Qwen3.8 review records differ from stage summary")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Qwen3.8 review records contain duplicate samples")
    return records, summary


def _load_annotations(output_root: Path) -> list[Qwen38HumanReviewAnnotation]:
    path = output_root / "human_review/annotations.jsonl"
    if not path.is_file():
        return []
    annotations = [
        Qwen38HumanReviewAnnotation.model_validate(value)
        for value in _read_jsonl(path)
    ]
    keys = [(item.sample_id, item.request_fingerprint) for item in annotations]
    if len(keys) != len(set(keys)):
        raise ValueError("human review annotations contain duplicate sample fingerprints")
    return annotations


def _current_annotation_map(
    records: list[Qwen38RecaptionRecord],
    annotations: list[Qwen38HumanReviewAnnotation],
) -> dict[str, Qwen38HumanReviewAnnotation]:
    current_fingerprints = {
        record.sample_id: record.request_fingerprint
        for record in records
        if record.request_fingerprint is not None
    }
    return {
        item.sample_id: item
        for item in annotations
        if current_fingerprints.get(item.sample_id) == item.request_fingerprint
    }


def _build_summary(
    *,
    source_output_root: Path,
    source_summary_sha256: str,
    records: list[Qwen38RecaptionRecord],
    stage_summary: Qwen38RecaptionSummary,
    annotations: list[Qwen38HumanReviewAnnotation],
    generated_at: datetime,
) -> Qwen38HumanReviewSummary:
    current = _current_annotation_map(records, annotations)
    decisions = Counter(item.decision for item in current.values())
    severities = Counter(
        item.severity for item in current.values() if item.severity is not None
    )
    issue_tags = Counter(tag for item in current.values() for tag in item.issue_tags)
    reviewed = len(current)
    unique_clips = len({record.clip_uid for record in records})
    ready = sum(record.status == "ready" for record in records)
    return Qwen38HumanReviewSummary(
        batch={
            "source_output_root": str(source_output_root),
            "source_summary_sha256": source_summary_sha256,
            "prompt_version": QWEN38_RECAPTION_PROMPT_VERSION,
            "policy_version": QWEN38_RECAPTION_POLICY_VERSION,
            "materializer_version": QWEN38_RECAPTION_MATERIALIZER_VERSION,
            "backend": stage_summary.backend_provenance.backend,
            "model": stage_summary.backend_provenance.served_model_name,
            "audio_semantics_root": stage_summary.audio_semantics_root,
            "audio_semantics_sha256": stage_summary.audio_semantics_records_sha256,
        },
        scope={
            "total_sample_count": len(records),
            "unique_clip_count": unique_clips,
            "conditioning_variants": dict(
                sorted(Counter(record.conditioning_variant for record in records).items())
            ),
            "inventory_scope": "current_h3_samples_inventory_only",
            "canonical_wide_coverage": False,
        },
        model={
            "ready": ready,
            "failed": sum(record.status == "failed" for record in records),
            "unsupported": sum(record.status == "unsupported" for record in records),
            "first_pass_ready": sum(
                record.status == "ready" and record.model_call_count == 1
                for record in records
            ),
            "repaired_ready": sum(
                record.status == "ready" and record.model_call_count > 1
                for record in records
            ),
        },
        human_review={
            "reviewed": reviewed,
            "unreviewed": len(records) - reviewed,
            "completion_percentage": (
                0.0 if not records else round(reviewed * 100.0 / len(records), 6)
            ),
            "pass": decisions["pass"],
            "issue": decisions["issue"],
            "skip": decisions["skip"],
            "severity_counts": dict(sorted(severities.items())),
            "issue_tag_counts": dict(sorted(issue_tags.items())),
            "stale_annotation_count": len(annotations) - reviewed,
        },
        generated_at=generated_at,
    )


def _annotations_csv(annotations: list[Qwen38HumanReviewAnnotation]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "schema_version",
            "sample_id",
            "clip_uid",
            "request_fingerprint",
            "decision",
            "severity",
            "issue_tags",
            "notes",
            "reviewed_at",
        )
    )
    for item in annotations:
        writer.writerow(
            (
                item.schema_version,
                item.sample_id,
                item.clip_uid,
                item.request_fingerprint,
                item.decision,
                item.severity or "",
                ";".join(item.issue_tags),
                item.notes,
                item.reviewed_at.isoformat(),
            )
        )
    return output.getvalue()


def _report_markdown(summary: Qwen38HumanReviewSummary) -> str:
    scope = summary.scope
    model = summary.model
    review = summary.human_review
    tags = review["issue_tag_counts"]
    common = sorted(tags.items(), key=lambda item: (-item[1], item[0]))[:5]
    common_text = ", ".join(f"{name}={count}" for name, count in common) or "none"
    major_blocker = sum(
        int(review["severity_counts"].get(key, 0)) for key in ("major", "blocker")
    )
    ready_rate = (
        0.0
        if not scope["total_sample_count"]
        else round(model["ready"] * 100.0 / scope["total_sample_count"], 2)
    )
    return (
        "# H3 Full Review Progress\n\n"
        f"- current H3 inventory: {scope['total_sample_count']} samples / "
        f"{scope['unique_clip_count']} unique target clips\n"
        f"- reviewed: {review['reviewed']} ({review['completion_percentage']}%)\n"
        f"- pass: {review['pass']}\n"
        f"- issue: {review['issue']}\n"
        f"- skip: {review['skip']}\n"
        f"- major/blocker: {major_blocker}\n"
        f"- most common issue categories: {common_text}\n"
        f"- model ready rate: {ready_rate}%\n"
        "- scope note: current H3 pair-rooted inventory only; Issue #11 "
        "canonical-wide migration pending\n"
    )


def _write_review_outputs(
    *,
    data_root: Path,
    source_output_root: Path,
    records: list[Qwen38RecaptionRecord],
    stage_summary: Qwen38RecaptionSummary,
    annotations: list[Qwen38HumanReviewAnnotation],
    generated_at: datetime,
) -> Qwen38HumanReviewSummary:
    ordered = sorted(
        annotations,
        key=lambda item: (item.sample_id, item.request_fingerprint),
    )
    summary = _build_summary(
        source_output_root=source_output_root,
        source_summary_sha256=_sha256_file(data_root / "summary.json"),
        records=records,
        stage_summary=stage_summary,
        annotations=ordered,
        generated_at=generated_at,
    )
    review_root = data_root / "human_review"
    _atomic_write_text(
        review_root / "annotations.jsonl",
        "".join(_compact_json(item.model_dump(mode="json")) + "\n" for item in ordered),
    )
    _atomic_write_text(review_root / "annotations.csv", _annotations_csv(ordered))
    _atomic_write_text(
        review_root / "summary.json",
        json.dumps(
            summary.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )
    _atomic_write_text(review_root / "report.md", _report_markdown(summary))
    return summary


def initialize_human_review(
    *,
    data_root: Path,
    source_output_root: Path,
) -> Qwen38HumanReviewSummary:
    records, stage_summary = _load_output(data_root)
    return _write_review_outputs(
        data_root=data_root,
        source_output_root=source_output_root,
        records=records,
        stage_summary=stage_summary,
        annotations=[],
        generated_at=datetime.now(UTC),
    )


def save_human_review(
    output_root: Path,
    payload: dict[str, object],
    *,
    reviewed_at: datetime | None = None,
) -> tuple[Qwen38HumanReviewAnnotation, Qwen38HumanReviewSummary]:
    root = output_root.expanduser().resolve(strict=True)
    records, stage_summary = _load_output(root)
    record_by_id = {record.sample_id: record for record in records}
    sample_id = payload.get("sample_id")
    if not isinstance(sample_id, str) or sample_id not in record_by_id:
        raise ValueError("human review references an unknown sample")
    record = record_by_id[sample_id]
    if record.request_fingerprint is None:
        raise ValueError("sample has no current request fingerprint")
    if payload.get("clip_uid") != record.clip_uid:
        raise ValueError("human review clip identity differs from current record")
    if payload.get("request_fingerprint") != record.request_fingerprint:
        raise ValueError("human review request fingerprint is stale")
    annotation = Qwen38HumanReviewAnnotation.model_validate(
        {
            **payload,
            "schema_version": HUMAN_REVIEW_VERSION,
            "reviewed_at": reviewed_at or datetime.now(UTC),
        }
    )
    existing = _load_annotations(root)
    by_key = {
        (item.sample_id, item.request_fingerprint): item for item in existing
    }
    by_key[(annotation.sample_id, annotation.request_fingerprint)] = annotation
    summary = _write_review_outputs(
        data_root=root,
        source_output_root=root,
        records=records,
        stage_summary=stage_summary,
        annotations=list(by_key.values()),
        generated_at=annotation.reviewed_at,
    )
    return annotation, summary


def current_human_reviews(output_root: Path) -> dict[str, object]:
    root = output_root.expanduser().resolve(strict=True)
    records, _ = _load_output(root)
    annotations = _load_annotations(root)
    current = _current_annotation_map(records, annotations)
    return {
        "annotations": [
            current[record.sample_id].model_dump(mode="json")
            for record in records
            if record.sample_id in current
        ],
        "stale_annotation_count": len(annotations) - len(current),
    }


def current_human_review_summary(output_root: Path) -> dict[str, object]:
    root = output_root.expanduser().resolve(strict=True)
    records, stage_summary = _load_output(root)
    summary = _build_summary(
        source_output_root=root,
        source_summary_sha256=_sha256_file(root / "summary.json"),
        records=records,
        stage_summary=stage_summary,
        annotations=_load_annotations(root),
        generated_at=datetime.now(UTC),
    )
    return summary.model_dump(mode="json")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source.resolve(strict=True))
    except OSError:
        shutil.copy2(source, destination)


def materialize_review_media(
    root: Path,
    records: list[Qwen38RecaptionRecord],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for record in records:
        sample_key = _sha256_text(record.sample_id)
        case_root = root / "media" / sample_key
        target = Path(record.target_video_path).expanduser().resolve(strict=True)
        if _sha256_file(target) != record.target_video_sha256:
            raise ValueError("review target video differs from record SHA")
        target_name = (
            f"target-{record.target_video_sha256[:12]}" + (target.suffix or ".mp4")
        )
        _link_or_copy(target, case_root / target_name)
        media: dict[str, object] = {
            "target": (
                f"media/{sample_key}/{target_name}?v={record.target_video_sha256}"
            ),
            "pictures": {},
            "audios": {},
        }
        if record.reference_contract is not None:
            picture_paths: dict[int, str] = {}
            for picture in record.reference_contract.pictures:
                source = Path(picture.image_path).expanduser().resolve(strict=True)
                if _sha256_file(source) != picture.image_sha256:
                    raise ValueError("review Picture differs from record SHA")
                name = (
                    f"picture-{picture.image_index}-{picture.image_sha256[:12]}"
                    + (source.suffix or ".png")
                )
                _link_or_copy(source, case_root / name)
                picture_paths[picture.image_index] = (
                    f"media/{sample_key}/{name}?v={picture.image_sha256}"
                )
            audio_paths: dict[int, str] = {}
            for audio in record.reference_contract.audios:
                source = Path(audio.path).expanduser().resolve(strict=True)
                if _sha256_file(source) != audio.sha256:
                    raise ValueError("review Audio differs from record SHA")
                name = (
                    f"audio-{audio.audio_index}-{audio.sha256[:12]}"
                    + (source.suffix or ".flac")
                )
                _link_or_copy(source, case_root / name)
                audio_paths[audio.audio_index] = (
                    f"media/{sample_key}/{name}?v={audio.sha256}"
                )
            media["pictures"] = picture_paths
            media["audios"] = audio_paths
        result[record.sample_id] = media
    return result


def _json_pre(value: object) -> str:
    return html.escape(json.dumps(value, ensure_ascii=False, indent=2))


def render_review_html(
    records: list[Qwen38RecaptionRecord],
    media_by_sample: dict[str, dict[str, object]],
) -> str:
    cards: list[str] = []
    for index, record in enumerate(records):
        media = media_by_sample[record.sample_id]
        picture_media = media["pictures"]
        audio_media = media["audios"]
        pictures = ""
        audios = ""
        if record.reference_contract is not None:
            pictures = "".join(
                "<figure><img loading='lazy' src='"
                + html.escape(picture_media[item.image_index], quote=True)
                + "'><figcaption><b>"
                + html.escape(item.picture_label)
                + "</b><br>kind="
                + html.escape(item.kind)
                + "<br>entity_id="
                + html.escape(str(item.entity_id))
                + "<br>owner_entity_id="
                + html.escape(str(item.owner_entity_id))
                + "<br>attribute_type="
                + html.escape(str(item.attribute_type))
                + "<br>source="
                + html.escape(item.image_path)
                + "<br>sha="
                + item.image_sha256[:12]
                + "</figcaption></figure>"
                for item in record.reference_contract.pictures
            )
            audios = "".join(
                "<figure><audio controls preload='none' src='"
                + html.escape(audio_media[item.audio_index], quote=True)
                + "'></audio><figcaption><b>"
                + html.escape(item.audio_label)
                + "</b><br>kind="
                + html.escape(item.kind)
                + "<br>subject="
                + html.escape(str(item.subject_label))
                + "<br>entity_id="
                + html.escape(str(item.entity_id))
                + "<br>speaker_id="
                + html.escape(str(item.speaker_id))
                + "<br>retention="
                + html.escape(item.retention_marker)
                + "<br>source="
                + html.escape(item.path)
                + "<br>sha="
                + item.sha256[:12]
                + "</figcaption></figure>"
                for item in record.reference_contract.audios
            )
        audit_items = (
            []
            if record.structured_h3_sections is None
            else record.structured_h3_sections.audio_fact_audit
        )
        audit_counts = Counter(item.action for item in audit_items)
        audit_summary = (
            f"{audit_counts['preserved']} preserved · "
            f"{audit_counts['attribution_generalized']} attribution generalized"
        )
        raw_name = f"{index:04d}-{_sha256_text(record.sample_id)[:12]}.json"
        request_short = (
            "unavailable"
            if record.request_fingerprint is None
            else record.request_fingerprint[:12]
        )
        checkboxes = "".join(
            "<label><input type='checkbox' name='issue_tag' value='"
            + tag
            + "'>"
            + tag
            + "</label>"
            for tag in REVIEW_ISSUE_TAGS
        )
        facts = (
            "[unavailable]"
            if record.audio_facts is None
            else _json_pre(record.audio_facts.model_dump(mode="json"))
        )
        structured = (
            "[unavailable]"
            if record.structured_h3_sections is None
            else _json_pre(record.structured_h3_sections.model_dump(mode="json"))
        )
        failure = (
            "[none]"
            if record.failure is None
            else _json_pre(record.failure.model_dump(mode="json"))
        )
        cards.append(
            "<article class='case' data-sample-id='"
            + html.escape(record.sample_id, quote=True)
            + "' data-model-status='"
            + record.status
            + "' data-review-decision='unreviewed' data-issue-tags=''>"
            + "<span hidden data-clip-uid='"
            + html.escape(record.clip_uid, quote=True)
            + "' data-request-fingerprint='"
            + html.escape(record.request_fingerprint or "", quote=True)
            + "'></span>"
            + "<h2>"
            + html.escape(record.sample_id)
            + "</h2><p><b>clip_uid:</b> "
            + html.escape(record.clip_uid)
            + "<br><b>conditioning_variant:</b> "
            + record.conditioning_variant
            + "<br><b>request fingerprint:</b> "
            + request_short
            + "<br><b>prompt version:</b> "
            + record.prompt_version
            + "<br><b>model status:</b> "
            + record.status
            + "<br><b>model calls:</b> "
            + str(record.model_call_count)
            + "<br><b>warnings:</b> "
            + html.escape(", ".join(record.validation_warnings) or "none")
            + "</p><h3>Target media</h3><video controls preload='metadata' src='"
            + html.escape(media["target"], quote=True)
            + "'></video><p>source="
            + html.escape(record.target_video_path)
            + "<br>sha="
            + record.target_video_sha256[:12]
            + "</p><h3>Visual references</h3><div class='media-row'>"
            + (pictures or "[none]")
            + "</div><h3>Audio references</h3><div class='media-row'>"
            + (audios or "[none]")
            + "</div><details><summary>Rough Visual instruction</summary><pre>"
            + html.escape(record.rough_r2v_instruction)
            + "</pre></details><details><summary>Reference contract</summary><pre>"
            + (
                "[unavailable]"
                if record.reference_contract is None
                else _json_pre(record.reference_contract.model_dump(mode="json"))
            )
            + "</pre></details><details><summary>Audio semantics and facts</summary><pre>"
            + facts
            + "</pre></details><details><summary>Audio fact audit: "
            + audit_summary
            + "</summary><pre>"
            + _json_pre([item.model_dump(mode="json") for item in audit_items])
            + "</pre></details><h3>Final output</h3><pre>"
            + html.escape(record.rendered_h3_prompt or "[unavailable]")
            + "</pre><details><summary>Structured output</summary><pre>"
            + structured
            + "</pre></details><details><summary>Failure/issues</summary><pre>"
            + failure
            + "</pre></details><details><summary>Raw responses</summary><p><a href='raw_responses/"
            + raw_name
            + "'>Open raw response JSON</a></p></details><form class='review-controls'>"
            + "<h3>Human review</h3><div class='decisions'>Decision: "
            + "<label><input type='radio' name='decision' value='pass'>Pass</label>"
            + "<label><input type='radio' name='decision' value='issue'>Issue</label>"
            + "<label><input type='radio' name='decision' value='skip'>Skip</label>"
            + "</div><label>Severity <select name='severity'><option value=''>none</option>"
            + "<option value='minor'>minor</option><option value='major'>major</option>"
            + "<option value='blocker'>blocker</option></select></label>"
            + "<div class='issue-tags'>Issue tags: "
            + checkboxes
            + "</div><label>Notes<textarea name='notes'></textarea></label>"
            + "<button type='button' class='save-review'>Save Review</button>"
            + "<span class='saved-at'></span></form></article>"
        )
    issue_options = "".join(
        f"<option value='{tag}'>{tag}</option>" for tag in REVIEW_ISSUE_TAGS
    )
    script = """
const cards = Array.from(document.querySelectorAll('.case'));
const summaryNode = document.getElementById('dashboard');
const issueFilter = document.getElementById('issue-filter');
function currentPayload(card) {
  const checked = card.querySelector('input[name="decision"]:checked');
  return {
    sample_id: card.dataset.sampleId,
    clip_uid: card.querySelector('[data-clip-uid]').dataset.clipUid,
    request_fingerprint: card.querySelector('[data-request-fingerprint]').dataset.requestFingerprint,
    decision: checked ? checked.value : null,
    severity: card.querySelector('select[name="severity"]').value || null,
    issue_tags: Array.from(card.querySelectorAll('input[name="issue_tag"]:checked')).map(node => node.value),
    notes: card.querySelector('textarea[name="notes"]').value
  };
}
function applyAnnotation(card, annotation) {
  const decision = card.querySelector(`input[name="decision"][value="${annotation.decision}"]`);
  if (decision) decision.checked = true;
  card.querySelector('select[name="severity"]').value = annotation.severity || '';
  card.querySelector('textarea[name="notes"]').value = annotation.notes || '';
  card.querySelectorAll('input[name="issue_tag"]').forEach(node => { node.checked = annotation.issue_tags.includes(node.value); });
  card.dataset.reviewDecision = annotation.decision;
  card.dataset.issueTags = annotation.issue_tags.join(',');
  card.querySelector('.saved-at').textContent = `Saved at ${annotation.reviewed_at}`;
}
function renderDashboard(summary) {
  const review = summary.human_review;
  summaryNode.textContent = `Reviewed ${review.reviewed} / ${summary.scope.total_sample_count} · Pass ${review.pass} · Issue ${review.issue} · Skip ${review.skip} · Unreviewed ${review.unreviewed} · Severity ${JSON.stringify(review.severity_counts)} · Tags ${JSON.stringify(review.issue_tag_counts)}`;
}
async function loadReviews() {
  const [reviewsResponse, summaryResponse] = await Promise.all([fetch('/api/reviews', {cache: 'no-store'}), fetch('/api/review-summary', {cache: 'no-store'})]);
  if (!reviewsResponse.ok || !summaryResponse.ok) throw new Error('review API unavailable');
  const reviews = await reviewsResponse.json();
  const summary = await summaryResponse.json();
  const bySample = new Map(reviews.annotations.map(item => [item.sample_id, item]));
  cards.forEach(card => { const item = bySample.get(card.dataset.sampleId); if (item) applyAnnotation(card, item); });
  renderDashboard(summary);
}
async function saveReview(card) {
  const payload = currentPayload(card);
  if (!payload.decision) { alert('Choose a decision first.'); return; }
  const response = await fetch('/api/review', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
  const result = await response.json();
  if (!response.ok) { alert(result.error || 'Review save failed.'); return; }
  applyAnnotation(card, result.annotation);
  renderDashboard(result.summary);
  applyFilters();
}
function applyFilters() {
  const decision = document.querySelector('input[name="review-filter"]:checked').value;
  const tag = issueFilter.value;
  cards.forEach(card => {
    const decisionMatch = decision === 'all' || (decision === 'model_failed' ? card.dataset.modelStatus === 'failed' : card.dataset.reviewDecision === decision);
    const tagMatch = !tag || card.dataset.issueTags.split(',').includes(tag);
    card.hidden = !(decisionMatch && tagMatch);
  });
}
document.querySelectorAll('.save-review').forEach(button => button.addEventListener('click', () => saveReview(button.closest('.case'))));
document.querySelectorAll('input[name="review-filter"]').forEach(node => node.addEventListener('change', applyFilters));
issueFilter.addEventListener('change', applyFilters);
loadReviews().catch(error => { summaryNode.textContent = `Review API unavailable: ${error.message}`; });
"""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Qwen3.8 H3 full review</title><style>"
        "body{font:14px system-ui;margin:20px;background:#f4f4f1;color:#171717}"
        "header{position:sticky;top:0;background:#fff;border:1px solid #bbb;padding:12px;z-index:3}"
        "article{background:#fff;border:1px solid #bbb;margin:18px 0;padding:16px;max-width:1280px}"
        "video{width:min(760px,100%)}.media-row{display:flex;gap:12px;overflow:auto}"
        "figure{margin:10px 0;min-width:260px}figure img{height:190px;max-width:320px;object-fit:contain;background:#eee}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f7f7;padding:10px}"
        ".review-controls{border-top:2px solid #777;padding-top:12px}.issue-tags{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:5px;margin:10px 0}"
        "textarea{display:block;width:min(900px,95%);min-height:80px}.saved-at{margin-left:10px;color:#176b31}"
        "</style></head><body><header><h1>Qwen3.8 H3 full review</h1>"
        "<p>This batch covers the current H3 samples inventory only. It is not canonical-wide coverage.</p>"
        "<div id='dashboard'>Loading review progress…</div><fieldset><legend>Filter</legend>"
        "<label><input type='radio' name='review-filter' value='all' checked>All</label>"
        "<label><input type='radio' name='review-filter' value='unreviewed'>Unreviewed</label>"
        "<label><input type='radio' name='review-filter' value='pass'>Pass</label>"
        "<label><input type='radio' name='review-filter' value='issue'>Issue</label>"
        "<label><input type='radio' name='review-filter' value='skip'>Skip</label>"
        "<label><input type='radio' name='review-filter' value='model_failed'>Model failed</label>"
        "<label>Issue tag <select id='issue-filter'><option value=''>all</option>"
        + issue_options
        + "</select></label></fieldset></header><main>"
        + "".join(cards)
        + "</main><script>"
        + script
        + "</script></body></html>"
    )


def _json_response(handler: BaseHTTPRequestHandler, status: HTTPStatus, value: object) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
    handler.end_headers()
    handler.wfile.write(payload)


def make_review_handler(output_root: Path) -> type[BaseHTTPRequestHandler]:
    root = output_root.expanduser().resolve(strict=True)

    class ReviewHandler(BaseHTTPRequestHandler):
        def _send_file(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            request_path = unquote(urlsplit(self.path).path)
            if request_path in {"/", "/review.html"}:
                self._send_file(root / "review.html")
                return
            if request_path == "/api/reviews":
                _json_response(self, HTTPStatus.OK, current_human_reviews(root))
                return
            if request_path == "/api/review-summary":
                _json_response(self, HTTPStatus.OK, current_human_review_summary(root))
                return
            relative = PurePosixPath(request_path.removeprefix("/"))
            if not relative.parts or relative.parts[0] not in {"media", "raw_responses"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if any(part in {"", ".", ".."} for part in relative.parts):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            candidate = root.joinpath(*relative.parts)
            self._send_file(candidate)

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/review":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1024 * 1024:
                    raise ValueError("invalid review request length")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise TypeError("review request must be a JSON object")
                annotation, summary = save_human_review(root, value)
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
    "HUMAN_REVIEW_VERSION",
    "REVIEW_DECISIONS",
    "REVIEW_ISSUE_TAGS",
    "REVIEW_SEVERITIES",
    "Qwen38HumanReviewAnnotation",
    "Qwen38HumanReviewSummary",
    "current_human_review_summary",
    "current_human_reviews",
    "initialize_human_review",
    "make_review_handler",
    "materialize_review_media",
    "render_review_html",
    "save_human_review",
]
