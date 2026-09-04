from __future__ import annotations

import csv
import hashlib
import html
import json
import mimetypes
import os
import re
import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, ValidationError, model_validator

from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoClipJob,
    MimoInventory,
    MimoRawResponse,
    MimoRecord,
)
from r2v_data_v2.h3.mimo25_h3_materializer import (
    MimoH3ShadowRecord,
    MimoH3ShadowSummary,
)
from r2v_data_v2.h3.qwen38_h3_recaption import Qwen38RecaptionRecord
from r2v_data_v2.h3.schemas import SchemaModel

MIMO25_REVIEW_ANNOTATION_VERSION = "r2v.h3.mimo25_human_review.1"
MIMO25_REVIEW_SUMMARY_VERSION = "r2v.h3.mimo25_human_review_summary.1"

Decision = Literal["PASS", "ISSUE", "SKIP"]
IssueTag = Literal[
    "speaker_grouping_issue",
    "entity_binding_issue",
    "segment_composition_issue",
    "missed_speaker_change",
    "false_speaker_change",
    "audio_event_oversegmentation",
    "audio_hallucination",
    "missed_audio_event",
    "music_issue",
    "soundscape_issue",
    "visual_caption_issue",
    "h3_format_issue",
    "asr_conflict_warning",
    "other",
]
ISSUE_TAGS = (
    "speaker_grouping_issue",
    "entity_binding_issue",
    "segment_composition_issue",
    "missed_speaker_change",
    "false_speaker_change",
    "audio_event_oversegmentation",
    "audio_hallucination",
    "missed_audio_event",
    "music_issue",
    "soundscape_issue",
    "visual_caption_issue",
    "h3_format_issue",
    "asr_conflict_warning",
    "other",
)


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


class MimoHumanReviewAnnotation(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_human_review.1"] = (
        MIMO25_REVIEW_ANNOTATION_VERSION
    )
    clip_uid: str
    record_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Decision
    issue_tags: list[IssueTag]
    notes: str
    reviewed_at: str

    @model_validator(mode="after")
    def validate_annotation(self) -> MimoHumanReviewAnnotation:
        if self.decision == "ISSUE" and not self.issue_tags:
            raise ValueError("ISSUE review requires at least one issue tag")
        if self.decision != "ISSUE" and self.issue_tags:
            raise ValueError("only ISSUE review may publish issue tags")
        if len(self.issue_tags) != len(set(self.issue_tags)):
            raise ValueError("MiMo review issue tags must be unique")
        return self


class MimoHumanReviewSummary(SchemaModel):
    schema_version: Literal["r2v.h3.mimo25_human_review_summary.1"] = (
        MIMO25_REVIEW_SUMMARY_VERSION
    )
    current_case_count: int = Field(ge=0)
    reviewed_count: int = Field(ge=0)
    unreviewed_count: int = Field(ge=0)
    decision_counts: dict[str, int]
    issue_tag_counts: dict[str, int]
    stale_annotation_count: int = Field(ge=0)


@dataclass(frozen=True)
class MimoReviewCase:
    clip_uid: str
    record_fingerprint: str
    payload: dict[str, object]


def _token(path: Path) -> str:
    return _sha256_text(str(path))[:24]


def _review_case_fingerprint(
    record: MimoRecord,
    variants: Sequence[MimoH3ShadowRecord],
) -> str:
    return _sha256_text(
        _compact_json(
            {
                "mimo_record_fingerprint": record.record_fingerprint,
                "shadow_record_fingerprints": [
                    [item.sample_id, item.record_fingerprint]
                    for item in sorted(variants, key=lambda row: row.sample_id)
                ],
            }
        )
    )


def _reference_selection_audit(job: MimoClipJob) -> list[dict[str, object]]:
    selected_by_source_index = {
        item.source_image_index: item for item in job.reference_images
    }
    dropped_by_source_index = {
        item.source_image_index: item
        for item in job.reference_selection.dropped_references
    }
    rows: list[dict[str, object]] = []
    for source_index in range(1, job.reference_selection.original_picture_count + 1):
        selected = selected_by_source_index.get(source_index)
        dropped = dropped_by_source_index.get(source_index)
        if selected is not None:
            rows.append(
                {
                    "source_image_index": selected.source_image_index,
                    "source_image_label": selected.source_image_label,
                    "source_image_id": selected.source_image_id,
                    "picture_label": selected.picture_label,
                    "kind": selected.kind,
                    "entity_id": selected.entity_id,
                    "owner_entity_id": selected.owner_entity_id,
                    "attribute_type": selected.attribute_type,
                    "selected": True,
                    "drop_reason": None,
                }
            )
        elif dropped is not None:
            rows.append(
                {
                    **dropped.model_dump(mode="json"),
                    "picture_label": None,
                    "selected": False,
                }
            )
        else:  # pragma: no cover - MimoReferenceSelection validates coverage
            raise ValueError("MiMo review reference selection inventory differs")
    return rows


def build_review_cases(
    *,
    mimo_root: Path,
    shadow_root: Path,
    legacy_qwen38_root: Path | None = None,
) -> tuple[list[MimoReviewCase], dict[str, Path]]:
    mimo = mimo_root.expanduser().resolve(strict=True)
    shadow = shadow_root.expanduser().resolve(strict=True)
    inventory = MimoInventory.model_validate_json(
        (mimo / "inventory.json").read_text(encoding="utf-8")
    )
    records = [
        MimoRecord.model_validate(row) for row in _read_jsonl(mimo / "records.jsonl")
    ]
    shadow_records = [
        MimoH3ShadowRecord.model_validate(row)
        for row in _read_jsonl(shadow / "records.jsonl")
    ]
    shadow_summary_path = shadow / "summary.json"
    if shadow_summary_path.is_file():
        try:
            shadow_summary = MimoH3ShadowSummary.model_validate_json(
                shadow_summary_path.read_text(encoding="utf-8")
            )
        except ValidationError as exc:
            raise ValueError(
                "MiMo review shadow provenance differs from current AV annotation"
            ) from exc
        if (
            shadow_summary.source_mimo_inventory_fingerprint
            != inventory.inventory_fingerprint
            or shadow_summary.sample_count != len(shadow_records)
        ):
            raise ValueError(
                "MiMo review shadow provenance differs from current AV annotation"
            )
    shadow_by_clip: dict[str, list[MimoH3ShadowRecord]] = {}
    for record in shadow_records:
        shadow_by_clip.setdefault(record.clip_uid, []).append(record)
    legacy_by_sample: dict[str, Qwen38RecaptionRecord] = {}
    if legacy_qwen38_root is not None:
        legacy_path = (
            legacy_qwen38_root.expanduser().resolve(strict=True) / "records.jsonl"
        )
        legacy_by_sample = {
            item.sample_id: item
            for item in (
                Qwen38RecaptionRecord.model_validate(row)
                for row in _read_jsonl(legacy_path)
            )
        }
    record_by_clip = {item.clip_uid: item for item in records}
    job_clip_ids = {item.clip_uid for item in inventory.jobs}
    if len(record_by_clip) != len(records) or set(record_by_clip) != job_clip_ids:
        raise ValueError("MiMo review records do not exactly cover current inventory")
    if set(shadow_by_clip) - job_clip_ids:
        raise ValueError(
            "MiMo review shadow provenance differs from current AV annotation"
        )
    media: dict[str, Path] = {}
    cases: list[MimoReviewCase] = []
    for job in inventory.jobs:
        record = record_by_clip.get(job.clip_uid)
        if record is None:
            raise ValueError(f"MiMo review record is missing: {job.clip_uid}")
        if (
            record.inventory_fingerprint != inventory.inventory_fingerprint
            or record.request_fingerprint != job.request_fingerprint
        ):
            raise ValueError("MiMo review AV annotation provenance mismatch")
        variants = sorted(
            shadow_by_clip.get(job.clip_uid, []), key=lambda item: item.sample_id
        )
        source_variants = [
            item for item in variants if item.derived_from_pair_type is None
        ]
        actual_sample_ids = [item.source_h3_sample_id for item in source_variants]
        if actual_sample_ids != sorted(job.source_h3_sample_ids):
            raise ValueError(
                "MiMo review shadow provenance differs from current AV annotation"
            )
        if any(
            item.clip_uid != job.clip_uid
            or item.source_mimo_record_fingerprint != record.record_fingerprint
            or (
                item.derived_from_pair_type is not None
                and (
                    item.derived_from_pair_type != "canonical"
                    or item.pair_type not in {"canonical", "in_pair"}
                    or item.source_h3_sample_id not in job.source_h3_sample_ids
                )
            )
            for item in variants
        ):
            raise ValueError(
                "MiMo review shadow provenance differs from current AV annotation"
            )
        raw = MimoRawResponse.model_validate_json(
            (mimo / "raw_responses" / f"{job.clip_uid}.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            raw.clip_uid != job.clip_uid
            or raw.request_fingerprint != job.request_fingerprint
        ):
            raise ValueError("MiMo review runtime diagnostics provenance mismatch")
        target = Path(job.target_video_path).resolve(strict=True)
        media[_token(target)] = target
        references = []
        for item in job.reference_images:
            path = Path(item.image_artifact_path).resolve(strict=True)
            token = _token(path)
            media[token] = path
            references.append(
                {**item.model_dump(mode="json"), "media_url": f"/media/{token}"}
            )
        shadow_variants = []
        for item in variants:
            variant = item.model_dump(mode="json")
            audio_references = []
            for audio in item.audio_references:
                path = Path(audio.audio_path).resolve(strict=True)
                if _sha256_file(path) != audio.audio_sha256:
                    raise ValueError("MiMo review Audio reference provenance mismatch")
                token = _token(path)
                media[token] = path
                audio_references.append(
                    {**audio.model_dump(mode="json"), "media_url": f"/media/{token}"}
                )
            variant["audio_references"] = audio_references
            shadow_variants.append(variant)
        review_fingerprint = _review_case_fingerprint(record, variants)
        annotation = record.annotation
        audio_by_segment = (
            {}
            if annotation is None
            else {
                item.segment_id: item
                for item in annotation.audio_observation.segment_decisions
            }
        )
        grounding_by_segment = (
            {}
            if annotation is None
            else {
                item.segment_id: item
                for item in annotation.av_grounding.segment_groundings
            }
        )
        view_by_segment = (
            {}
            if annotation is None
            else {
                item.segment_id: item
                for item in annotation.visual_observation.segment_views
            }
        )
        presentation_counts: Counter[str] = Counter()
        binding_change_counts: Counter[str] = Counter()
        segment_rows: list[dict[str, object]] = []
        for source in job.segments:
            audio = audio_by_segment.get(source.segment_id)
            grounding = grounding_by_segment.get(source.segment_id)
            view = view_by_segment.get(source.segment_id)
            selected_entity = None if grounding is None else grounding.entity_id
            if grounding is not None:
                presentation_counts[grounding.speech_presentation] += 1
            if source.current_entity_id is None and selected_entity is not None:
                binding_change = "entity_binding_added"
            elif source.current_entity_id is not None and selected_entity is None:
                binding_change = "entity_binding_removed"
            elif source.current_entity_id == selected_entity:
                binding_change = "entity_binding_preserved"
            else:
                binding_change = "entity_binding_changed"
            binding_change_counts[binding_change] += 1
            direct_anchor_override = (
                source.identity_scope == "direct_anchor_present"
                and source.current_entity_id != selected_entity
            )
            if direct_anchor_override:
                binding_change_counts["direct_anchor_override"] += 1
            segment_rows.append(
                {
                    **source.model_dump(mode="json"),
                    "stage_a_visible_entity_ids": (
                        [] if view is None else view.visible_entity_ids
                    ),
                    "stage_a_entity_observations": (
                        []
                        if view is None
                        else [
                            item.model_dump(mode="json")
                            for item in view.entity_observations
                        ]
                    ),
                    "stage_b_audio_decision": (
                        None if audio is None else audio.model_dump(mode="json")
                    ),
                    "stage_c_av_grounding": (
                        None
                        if grounding is None
                        else grounding.model_dump(mode="json")
                    ),
                    "binding_change": binding_change,
                    "direct_anchor_override": direct_anchor_override,
                }
            )
        payload = {
            "clip_uid": job.clip_uid,
            "review_case_fingerprint": review_fingerprint,
            "target_video_url": f"/media/{_token(target)}",
            "references": references,
            "reference_selection": job.reference_selection.model_dump(mode="json"),
            "reference_selection_audit": _reference_selection_audit(job),
            "source_segments": segment_rows,
            "mimo_record": record.model_dump(mode="json"),
            "visual_observation": (
                None
                if annotation is None
                else annotation.visual_observation.model_dump(mode="json")
            ),
            "audio_observation": (
                None
                if annotation is None
                else annotation.audio_observation.model_dump(mode="json")
            ),
            "av_grounding": (
                None
                if annotation is None
                else annotation.av_grounding.model_dump(mode="json")
            ),
            "presentation_counts": dict(sorted(presentation_counts.items())),
            "binding_change_counts": dict(sorted(binding_change_counts.items())),
            "mimo_runtime_diagnostics": raw.diagnostics,
            "shadow_variants": shadow_variants,
            "legacy_qwen38": {
                item.sample_id: legacy_by_sample[item.sample_id].rendered_h3_prompt
                for item in variants
                if item.sample_id in legacy_by_sample
            },
        }
        cases.append(
            MimoReviewCase(
                clip_uid=job.clip_uid,
                record_fingerprint=review_fingerprint,
                payload=payload,
            )
        )
    return cases, media


class MimoReviewStore:
    def __init__(self, root: Path, cases: Sequence[MimoReviewCase]) -> None:
        self.root = root.expanduser().resolve(strict=False)
        self.cases = list(cases)
        self.current = {item.clip_uid: item.record_fingerprint for item in cases}

    @property
    def annotations_path(self) -> Path:
        return self.root / "annotations.jsonl"

    def load_all(self) -> list[MimoHumanReviewAnnotation]:
        if not self.annotations_path.is_file():
            return []
        return [
            MimoHumanReviewAnnotation.model_validate(row)
            for row in _read_jsonl(self.annotations_path)
        ]

    def current_annotations(self) -> dict[str, MimoHumanReviewAnnotation]:
        return {
            item.clip_uid: item
            for item in self.load_all()
            if self.current.get(item.clip_uid) == item.record_fingerprint
        }

    def save(self, annotation: MimoHumanReviewAnnotation) -> MimoHumanReviewSummary:
        expected = self.current.get(annotation.clip_uid)
        if expected is None or expected != annotation.record_fingerprint:
            raise ValueError("MiMo review annotation is stale or unknown")
        by_clip = {item.clip_uid: item for item in self.load_all()}
        by_clip[annotation.clip_uid] = annotation
        rows = [by_clip[key] for key in sorted(by_clip)]
        _atomic_text(
            self.annotations_path,
            "".join(
                _compact_json(item.model_dump(mode="json")) + "\n" for item in rows
            ),
        )
        return self.publish_derived()

    def publish_derived(self) -> MimoHumanReviewSummary:
        all_rows = self.load_all()
        current = [
            item
            for item in all_rows
            if self.current.get(item.clip_uid) == item.record_fingerprint
        ]
        decision_counts = Counter(item.decision for item in current)
        issue_counts = Counter(tag for item in current for tag in item.issue_tags)
        summary = MimoHumanReviewSummary(
            current_case_count=len(self.cases),
            reviewed_count=len(current),
            unreviewed_count=len(self.cases) - len(current),
            decision_counts=dict(sorted(decision_counts.items())),
            issue_tag_counts=dict(sorted(issue_counts.items())),
            stale_annotation_count=len(all_rows) - len(current),
        )
        _atomic_text(
            self.root / "summary.json",
            json.dumps(
                summary.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
        csv_path = self.root / "annotations.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{csv_path.name}.", dir=csv_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    (
                        "clip_uid",
                        "record_fingerprint",
                        "decision",
                        "issue_tags",
                        "notes",
                        "reviewed_at",
                    )
                )
                for item in sorted(current, key=lambda row: row.clip_uid):
                    writer.writerow(
                        (
                            item.clip_uid,
                            item.record_fingerprint,
                            item.decision,
                            "|".join(item.issue_tags),
                            item.notes,
                            item.reviewed_at,
                        )
                    )
            Path(temporary).replace(csv_path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        return summary


def render_review_html(
    cases: Sequence[MimoReviewCase], annotations: dict[str, MimoHumanReviewAnnotation]
) -> str:
    payload = [
        {**item.payload, "review_case_fingerprint": item.record_fingerprint}
        for item in cases
    ]
    annotation_payload = {
        key: value.model_dump(mode="json") for key, value in annotations.items()
    }
    tags = "".join(
        f'<label><input type="checkbox" value="{html.escape(tag)}"> {html.escape(tag)}</label>'
        for tag in ISSUE_TAGS
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>MiMo AV Review</title>
<style>body{{font:14px system-ui;margin:0;background:#f4f5f7;color:#18202a}}header{{position:sticky;top:0;background:#fff;padding:12px 20px;border-bottom:1px solid #ccd2da;z-index:2}}main{{display:grid;grid-template-columns:minmax(440px,1fr) minmax(560px,1.35fr);gap:16px;padding:16px}}video{{width:100%;max-height:46vh;background:#000}}.panel{{background:#fff;border:1px solid #ccd2da;padding:12px;border-radius:6px;overflow:auto}}.refs{{display:flex;gap:8px;overflow:auto}}.refs img{{height:130px}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border:1px solid #d8dde4;padding:5px;vertical-align:top}}pre{{white-space:pre-wrap}}button{{padding:10px 18px;margin:4px;font-weight:700}}.pass{{background:#d7f5df}}.issue,.offscreen{{background:#ffe1dc}}.skip{{background:#eee}}.counts span{{display:inline-block;margin:2px 8px 2px 0;padding:3px 6px;border:1px solid #ccd2da}}#tags label{{display:inline-block;margin:4px 10px 4px 0}}</style></head>
<body><header><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><b id="progress"></b> <label>Presentation <select id="presentation-filter" onchange="drawSegments()"><option value="all">all</option><option>onscreen_spoken</option><option>offscreen_spoken</option><option>voice_over</option><option>message_voice_over</option><option>device_playback</option><option>uncertain</option></select></label></header>
<main><section class="panel"><h2 id="title"></h2><video id="video" controls></video><h3>Frozen references</h3><div id="refs" class="refs"></div><h3>Reference-selection provenance</h3><div id="reference-audit"></div><h3>Stage A visual observation</h3><pre id="visual-observation"></pre><h3>Stage B audio observation</h3><pre id="audio-observation"></pre><h3>Stage C AV grounding</h3><pre id="av-grounding"></pre><h3>MiMo runtime diagnostics</h3><pre id="diagnostics"></pre></section>
<section class="panel"><h3>Presentation and binding counts</h3><div id="case-counts" class="counts"></div><h3>Segments</h3><div id="segments"></div><h3>Final materialized H3 shadow</h3><div id="shadow"></div><h3>Legacy Qwen3.8</h3><div id="legacy"></div><hr><button class="pass" onclick="save('PASS')">PASS</button><button class="issue" onclick="save('ISSUE')">ISSUE</button><button class="skip" onclick="save('SKIP')">SKIP</button><div id="tags">{tags}</div><textarea id="notes" rows="4" style="width:100%" placeholder="notes"></textarea></section></main>
<script>
const cases={json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")};
const initial={json.dumps(annotation_payload, ensure_ascii=False).replace("</", "<\\/")};
let index=0;
function esc(v){{return String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));}}
function audioMeta(a){{if(a.role==='voice_reference')return `Subject ${{a.subject_index}} / ${{esc(a.entity_id)}} / ${{esc(a.speaker_id||'no Sx')}}`;if(a.role==='music_reference')return `music: ${{esc(a.music_description)}}<br>interval: ${{a.source_start_time}}-${{a.source_end_time}} / ${{esc(a.interval_provenance)}}`;return `complete canonical Audio / ${{esc(a.interval_provenance)}}`;}}
function countBadges(values){{return Object.entries(values||{{}}).map(([key,value])=>`<span><b>${{esc(key)}}</b>: ${{value}}</span>`).join('');}}
function referenceAudit(c){{const rows=c.reference_selection_audit.map(x=>`<tr><td>${{esc(x.source_image_label)}}<br>${{esc(x.source_image_id)}}</td><td>${{esc(x.picture_label||'[dropped]')}}</td><td>${{esc(x.kind)}}</td><td>${{esc(x.entity_id||x.owner_entity_id||'')}}</td><td>${{esc(x.attribute_type||'')}}</td><td>${{x.selected?'selected':`dropped: ${{esc(x.drop_reason)}}`}}</td></tr>`).join('');return `<div class="counts">original ${{c.reference_selection.original_picture_count}} / selected ${{c.reference_selection.selected_picture_count}}</div><table><tr><th>source</th><th>task-local</th><th>kind</th><th>entity/owner</th><th>attribute</th><th>result</th></tr>${{rows}}</table>`;}}
function drawSegments(){{const c=cases[index],filter=document.getElementById('presentation-filter').value;const rows=c.source_segments.filter(x=>filter==='all'||x.stage_c_av_grounding?.speech_presentation===filter).map(x=>{{const grounding=x.stage_c_av_grounding||{{}};const cls=grounding.speech_presentation==='offscreen_spoken'?' class="offscreen"':'';return `<tr${{cls}}><td>${{esc(x.segment_id)}}<br>${{x.start_time}}-${{x.end_time}}</td><td>[${{esc(x.asr_language)}}] ${{esc(x.asr_text)}}</td><td>${{esc(x.source_speaker_cluster_id)}} / ${{esc(x.current_entity_id)}}<br>${{esc(x.identity_scope)}} / direct=${{x.direct_anchor_seconds}}</td><td>${{esc((x.stage_a_visible_entity_ids||[]).join(', '))}}<pre>${{esc(JSON.stringify(x.stage_a_entity_observations||[],null,2))}}</pre></td><td>${{esc(x.stage_b_audio_decision?.primary_speaker_group)}}<pre>${{esc(JSON.stringify(x.stage_b_audio_decision||null,null,2))}}</pre></td><td>${{esc(grounding.entity_id)}} / ${{esc(grounding.binding_status)}} / ${{esc(grounding.speech_presentation)}}<br>${{esc((grounding.evidence_codes||[]).join(', '))}}<br><b>${{esc(x.binding_change)}}</b>${{x.direct_anchor_override?' / direct-anchor override':''}}</td></tr>`;}}).join('');document.getElementById('segments').innerHTML='<table><tr><th>ID/time</th><th>ASR exact</th><th>source evidence</th><th>Stage A visible</th><th>Stage B gN/audio</th><th>Stage C selected grounding</th></tr>'+rows+'</table>';}}
function draw(){{const c=cases[index];document.getElementById('progress').textContent=` ${{index+1}} / ${{cases.length}}`;document.getElementById('title').textContent=c.clip_uid;document.getElementById('video').src=c.target_video_url;document.getElementById('refs').innerHTML=c.references.map(x=>`<figure><img src="${{x.media_url}}"><figcaption>${{esc(x.source_image_label)}} → ${{esc(x.picture_label)}}<br>${{esc(x.kind)}} / ${{esc(x.entity_id||x.owner_entity_id||'')}}</figcaption></figure>`).join('');document.getElementById('reference-audit').innerHTML=referenceAudit(c);document.getElementById('visual-observation').textContent=JSON.stringify(c.visual_observation||null,null,2);document.getElementById('audio-observation').textContent=JSON.stringify(c.audio_observation||null,null,2);document.getElementById('av-grounding').textContent=JSON.stringify(c.av_grounding||null,null,2);document.getElementById('diagnostics').textContent=JSON.stringify(c.mimo_runtime_diagnostics||[],null,2);document.getElementById('case-counts').innerHTML='<b>presentations</b> '+countBadges(c.presentation_counts)+'<br><b>binding</b> '+countBadges(c.binding_change_counts);drawSegments();document.getElementById('shadow').innerHTML=c.shadow_variants.map(x=>`<h4>${{esc(x.sample_id)}} (${{esc(x.conditioning_variant)}})</h4>${{(x.audio_references||[]).map(a=>`<div><b>&lt;Audio ${{a.audio_index}}&gt; / ${{esc(a.role)}}</b><br>${{audioMeta(a)}}<br>source: ${{esc(a.source_type)}}${{a.source_segment_id?' / '+esc(a.source_segment_id):''}}<br><audio controls preload="none" src="${{a.media_url}}"></audio></div>`).join('')}}<pre>${{esc(x.rendered_h3_prompt||x.failure_reason)}}</pre>`).join('');document.getElementById('legacy').innerHTML=Object.entries(c.legacy_qwen38).map(([key,value])=>`<h4>${{esc(key)}}</h4><pre>${{esc(value)}}</pre>`).join('')||'[not supplied]';const annotation=initial[c.clip_uid];document.getElementById('notes').value=annotation?.notes||'';document.querySelectorAll('#tags input').forEach(element=>element.checked=!!annotation?.issue_tags?.includes(element.value));}}
function move(delta){{index=Math.max(0,Math.min(cases.length-1,index+delta));draw();}}
async function save(decision){{const c=cases[index],selectedTags=[...document.querySelectorAll('#tags input:checked')].map(x=>x.value);if(decision==='ISSUE'&&!selectedTags.length){{alert('Select an issue tag');return;}}const body={{schema_version:'{MIMO25_REVIEW_ANNOTATION_VERSION}',clip_uid:c.clip_uid,record_fingerprint:c.review_case_fingerprint,decision,issue_tags:decision==='ISSUE'?selectedTags:[],notes:document.getElementById('notes').value,reviewed_at:new Date().toISOString()}};const response=await fetch('/api/annotation',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}});if(!response.ok){{alert(await response.text());return;}}initial[c.clip_uid]=body;move(1);}}
draw();
</script></body></html>"""


def make_review_server(
    *,
    host: str,
    port: int,
    cases: Sequence[MimoReviewCase],
    media: dict[str, Path],
    store: MimoReviewStore,
) -> ThreadingHTTPServer:
    page = render_review_html(cases, store.current_annotations()).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path == "/":
                self._send(HTTPStatus.OK, page, "text/html; charset=utf-8")
                return
            match = re.fullmatch(r"/media/([0-9a-f]{24})", parsed.path)
            path = None if match is None else media.get(match.group(1))
            if path is None or not path.is_file():
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            status = HTTPStatus.OK
            if range_header:
                range_match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
                if range_match is None:
                    self._send(
                        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "text/plain"
                    )
                    return
                start = int(range_match.group(1))
                end = min(
                    size - 1,
                    int(range_match.group(2)) if range_match.group(2) else size - 1,
                )
                if start > end:
                    self._send(
                        HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, b"", "text/plain"
                    )
                    return
                status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            self.send_response(status)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as handle:
                handle.seek(start)
                self.wfile.write(handle.read(length))

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/api/annotation":
                self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64 * 1024:
                    raise ValueError("invalid annotation payload length")
                annotation = MimoHumanReviewAnnotation.model_validate_json(
                    self.rfile.read(length)
                )
                summary = store.save(annotation)
                self._send(
                    HTTPStatus.OK,
                    _compact_json(summary.model_dump(mode="json")).encode(),
                    "application/json",
                )
            except (OSError, ValueError) as exc:
                self._send(
                    HTTPStatus.BAD_REQUEST,
                    str(exc).encode(),
                    "text/plain; charset=utf-8",
                )

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


__all__ = [
    "ISSUE_TAGS",
    "MimoHumanReviewAnnotation",
    "MimoHumanReviewSummary",
    "MimoReviewCase",
    "MimoReviewStore",
    "build_review_cases",
    "make_review_server",
    "render_review_html",
]
