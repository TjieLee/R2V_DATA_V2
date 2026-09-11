"""Standalone frozen-record A/B review; no inference or materialization."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr

from r2v_data_v2.h3.jea_final_renderer import FinalVisualReference
from r2v_data_v2.h3.mimo25_av_reconcile import MimoCaseManifest
from r2v_data_v2.h3.qwen38_h3_recaption import build_reference_contract
from r2v_data_v2.h3.sam_audio_stem_shadow import sha256_file
from r2v_data_v2.h3.visual_production_source import load_visual_production_inventory

VERSION = "r2v.h3.ra2va_binding_ab_review.1"
LABELS = {
    "SAME": "Speaker binding quality is effectively unchanged",
    "NEW_BETTER": "No-LR-ASD result is clearly better",
    "NEW_WORSE": "No-LR-ASD loses or corrupts a correct old speaker binding",
    "BOTH_WRONG": "Both versions have meaningful speaker-binding errors",
    "FORMAT_ONLY": "Semantic/speaker binding is acceptable; failure/difference is only caption/marker formatting",
}


class _Projection(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _Grounding(_Projection):
    segment_id: StrictStr
    binding_status: StrictStr
    entity_id: StrictStr | None
    speech_presentation: StrictStr
    evidence_codes: list[StrictStr]


class _Decision(_Projection):
    segment_id: StrictStr
    primary_speaker_group: StrictStr | None


class _Groundings(_Projection):
    segment_groundings: list[_Grounding]


class _Decisions(_Projection):
    segment_decisions: list[_Decision]


class _Caption(_Projection):
    shot1_caption: StrictStr
    summary: StrictStr


class _Annotation(_Projection):
    av_grounding: _Groundings
    audio_observation: _Decisions
    h3_semantics: _Caption


class _Record(_Projection):
    schema_version: StrictStr
    clip_uid: StrictStr
    status: Literal["ready", "failed"]
    annotation: _Annotation | None
    failure_code: StrictStr | None
    failure_reason: StrictStr | None
    failure_issues: list[dict]
    model_call_count: int = Field(ge=0)
    record_fingerprint: StrictStr


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _unique(rows, key):
    result = {getattr(row, key): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key} in review input")
    return result


def _records(root: Path) -> dict[str, _Record]:
    rows = []
    for line in (root / "records.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        row = _Record.model_validate(raw)
        if not row.schema_version.startswith("r2v.h3.mimo25_stem_reconcile."):
            raise ValueError("expected frozen MiMo stem reconcile records")
        if row.record_fingerprint != _hash({k: v for k, v in raw.items() if k != "record_fingerprint"}):
            raise ValueError("MiMo review record fingerprint mismatch")
        rows.append(row)
    return _unique(rows, "clip_uid")


def _side(record: _Record) -> dict:
    annotation = record.annotation
    rows = {}
    if annotation is not None:
        decisions = _unique(annotation.audio_observation.segment_decisions, "segment_id")
        groundings = _unique(annotation.av_grounding.segment_groundings, "segment_id")
        for uid in dict.fromkeys([*decisions, *groundings]):
            grounding, decision = groundings.get(uid), decisions.get(uid)
            rows[uid] = {
                "segment_id": uid, "primary_speaker_group": decision.primary_speaker_group if decision else None,
                "grounding_present": grounding is not None, "decision_present": decision is not None,
                **(grounding.model_dump() if grounding else {
                    "binding_status": None, "entity_id": None, "speech_presentation": None, "evidence_codes": [],
                }),
            }
    caption = annotation.h3_semantics.shot1_caption if annotation else ""
    # Read-only excerpt: preserve each complete lead-in since the preceding dialogue.
    dialogue = []
    start = 0
    for match in re.finditer(r"<d>.*?</d>", caption, re.DOTALL):
        dialogue.append(caption[start:match.end()])
        start = match.end()
    return {
        **record.model_dump(exclude={"annotation"}), "segments": rows,
        "annotation_present": annotation is not None, "shot1_caption": caption,
        "summary": annotation.h3_semantics.summary if annotation else "",
        "dialogue": dialogue,
    }


def _overlap(a: Path, b: Path) -> bool:
    return a.is_relative_to(b) or b.is_relative_to(a)


def build_review(*, visual_production_root: Path, visual_runs_root: Path, case_manifest: Path,
                 legacy_mimo_root: Path, no_lrasd_mimo_root: Path, output_root: Path) -> dict:
    visual_production_root = visual_production_root.resolve(strict=True)
    visual_runs_root = visual_runs_root.resolve(strict=True)
    legacy, new = legacy_mimo_root.resolve(strict=True), no_lrasd_mimo_root.resolve(strict=True)
    manifest_path = case_manifest.resolve(strict=True)
    output = output_root.resolve()
    protected = [visual_production_root, visual_runs_root, legacy, new, manifest_path]
    # A MiMo stage can be deep inside Audio production; never publish inside it.
    for root in (legacy, new):
        protected.extend(p.parent for p in root.parents if p.name == "sam_audio_stem_shadow_v1")
    if any(_overlap(output, p) for p in protected):
        raise ValueError("review output overlaps source/production roots")
    if output.exists():
        raise FileExistsError(output)
    source_hashes = {str(p): sha256_file(p) for p in (
        manifest_path, legacy / "records.jsonl", new / "records.jsonl", visual_production_root / "samples.jsonl",
    )}
    manifest = MimoCaseManifest.model_validate_json(manifest_path.read_text())
    old_rows, new_rows = _records(legacy), _records(new)
    visual = load_visual_production_inventory(
        visual_production_root=visual_production_root, visual_runs_root=visual_runs_root,
    )
    clips = {c.identity.clip_uid: c for c in visual.canonical_clips}
    if len(clips) != len(visual.canonical_clips):
        raise ValueError("duplicate Visual clip")
    if any(uid not in rows for rows in (clips, old_rows, new_rows) for uid in manifest.clip_uids):
        raise ValueError("selected clip missing from A/B or Visual inventory")
    cases, media = [], {}

    def own_media(filename):
        path = Path(filename).resolve(strict=True)
        if not path.is_file() or _overlap(output, path):
            raise ValueError("review media/output collision")
        suffix = path.suffix.lower()
        if suffix not in {".mp4", ".webm", ".mov", ".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError(f"unsupported browser review media: {suffix}")
        digest = sha256_file(path)
        relative = f"media/{digest}{suffix}"
        media[path] = (relative, digest)
        return relative

    summary = {"schema_version": VERSION, "clip_count": len(manifest.clip_uids)}
    counts = {name: Counter() for name in ("legacy", "no_lrasd")}
    transitions, visible_changes = Counter(), Counter(same_entity_id=0, changed_entity_id=0)
    for uid in manifest.clip_uids:
        context = clips[uid]
        references = [FinalVisualReference.from_visual(r) for r in context.sample.references]
        # Reuse canonical reference-label semantics, never derive speaker bindings.
        contract = build_reference_contract(SimpleNamespace(visual_references=references), "visual_only")
        old, current = _side(old_rows[uid]), _side(new_rows[uid])
        aligned = []
        for sid in dict.fromkeys([*old["segments"], *current["segments"]]):
            a, b = old["segments"].get(sid), current["segments"].get(sid)
            changed = [k for k in ("binding_status", "entity_id", "speech_presentation") if a and b and a[k] != b[k]]
            aligned.append({"segment_id": sid, "old": a, "new": b, "changed_fields": changed})
            if a and b and a["grounding_present"] and b["grounding_present"]:
                transitions[f'{a["binding_status"]} -> {b["binding_status"]}'] += 1
                if a["binding_status"] == b["binding_status"] == "visible_entity":
                    visible_changes["same_entity_id" if a["entity_id"] == b["entity_id"] else "changed_entity_id"] += 1
        for name, side in (("legacy", old), ("no_lrasd", current)):
            counts[name][side["status"]] += 1
            for row in side["segments"].values():
                if row["grounding_present"]:
                    counts[name]["groundings"] += 1
                    counts[name][f'status:{row["binding_status"]}'] += 1
                    counts[name]["null_entity"] += int(row["entity_id"] is None)
        cases.append({
            "clip_uid": uid, "target_video": own_media(context.sample.target_video),
            "references": [{"picture_label": p.picture_label, "kind": p.kind, "entity_id": p.entity_id,
                            "owner_entity_id": p.owner_entity_id, "media": own_media(p.image_path)} for p in contract.pictures],
            "subjects": [s.model_dump(mode="json") for s in contract.subjects],
            "old": old, "new": current, "aligned_segments": aligned,
            "segment_inventory_diff": set(old["segments"]) != set(current["segments"]),
        })
    for name, counter in counts.items():
        for field in ("ready", "failed", "null_entity"):
            summary[f"{name}_{field}_count"] = counter[field]
        summary[f"{name}_grounding_count"] = counter["groundings"]
        summary[f"{name}_binding_status_counts"] = {k[7:]: v for k, v in sorted(counter.items()) if k.startswith("status:")}
        for status in ("visible_entity", "offscreen"):
            summary[f"{name}_{status}_count"] = counter[f"status:{status}"]
    summary.update(transition_counts=dict(sorted(transitions.items())), visible_entity_transitions=dict(visible_changes),
                   model_calls_invoked=0, production_artifacts_modified=False)
    data = {"schema_version": VERSION, "legacy_root": str(legacy), "no_lrasd_root": str(new),
            "clip_count": len(cases), "labels": LABELS, "summary": summary, "cases": cases,
            "source_hashes": source_hashes, "media_hashes": {str(p): h for p, (_, h) in media.items()}}
    data["review_fingerprint"] = _hash(data)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".binding-ab-", dir=output.parent) as temporary:
        stage = Path(temporary) / "review"
        (stage / "media").mkdir(parents=True)
        for path, (relative, digest) in media.items():
            shutil.copyfile(path, stage / relative)
            if sha256_file(stage / relative) != digest:
                raise ValueError("review media changed during copy")
        for name, value in (("data.json", data), ("summary.json", summary)):
            (stage / name).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        shutil.copyfile(Path(__file__).with_suffix(".html"), stage / "review.html")
        if any(sha256_file(Path(p)) != h for p, h in source_hashes.items()):
            raise ValueError("review input changed during build")
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)
    return summary
