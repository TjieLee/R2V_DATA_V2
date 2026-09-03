from __future__ import annotations

import hashlib
import html
import json
import shutil
import uuid
from pathlib import Path
from typing import Literal

from pydantic import ConfigDict, Field

from r2v_data_v2.h3.qwen38_h3_recaption import (
    Qwen38RecaptionManifestCase,
    RecaptionReferenceContract,
)
from r2v_data_v2.h3.qwen_speech_presentation_recaption import (
    QwenPresentationRecord,
    QwenPresentationSummary,
)
from r2v_data_v2.h3.schemas import SchemaModel

QWEN_PRESENTATION_AB_COMPARISON_VERSION = (
    "r2v.h3.qwen_speech_presentation_ab_comparison.1"
)
QWEN_PRESENTATION_AB_SUMMARY_VERSION = (
    "r2v.h3.qwen_speech_presentation_ab_summary.1"
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
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


class _OldQwenRecordProjection(SchemaModel):
    model_config = ConfigDict(extra="ignore")

    sample_id: str
    clip_uid: str
    status: Literal["ready", "failed", "unsupported"]
    target_video_path: str
    target_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_contract: RecaptionReferenceContract | None = None
    rendered_h3_prompt: str | None = None


class QwenPresentationABComparison(SchemaModel):
    schema_version: Literal[
        "r2v.h3.qwen_speech_presentation_ab_comparison.1"
    ] = QWEN_PRESENTATION_AB_COMPARISON_VERSION
    sample_id: str
    clip_uid: str
    speech_decisions: dict[str, list[dict[str, object]]]
    qwen35_old_prompt: str | None
    qwen35_new_prompt: str | None
    qwen38_old_prompt: str | None
    qwen38_new_prompt: str | None


class QwenPresentationABSummary(SchemaModel):
    schema_version: Literal["r2v.h3.qwen_speech_presentation_ab_summary.1"] = (
        QWEN_PRESENTATION_AB_SUMMARY_VERSION
    )
    case_count: int = Field(ge=1)
    manifest_sha256_by_root: dict[str, str]
    qwen35_model_name: str
    qwen38_model_name: str
    qwen35: dict[str, object]
    qwen38: dict[str, object]


def _manifest(root: Path) -> tuple[list[Qwen38RecaptionManifestCase], str]:
    path = root / "manifest.jsonl"
    if not path.is_file():
        raise ValueError(f"Qwen comparison root has no manifest.jsonl: {root}")
    cases = [Qwen38RecaptionManifestCase.model_validate(row) for row in _read_jsonl(path)]
    if not cases:
        raise ValueError("Qwen comparison manifest is empty")
    sample_ids = [item.sample_id for item in cases]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Qwen comparison manifest contains duplicate sample IDs")
    return cases, _sha256_file(path)


def _old_output(
    root: Path,
) -> tuple[list[_OldQwenRecordProjection], str]:
    records = [
        _OldQwenRecordProjection.model_validate(row)
        for row in _read_jsonl(root / "records.jsonl")
    ]
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise TypeError("old Qwen summary must be a JSON object")
    provenance = summary.get("backend_provenance")
    if not isinstance(provenance, dict):
        raise TypeError("old Qwen summary backend provenance must be an object")
    model_name = provenance.get("served_model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("old Qwen summary lacks served model name")
    return records, model_name


def _new_output(
    root: Path,
) -> tuple[list[QwenPresentationRecord], QwenPresentationSummary]:
    records = [
        QwenPresentationRecord.model_validate(row)
        for row in _read_jsonl(root / "records.jsonl")
    ]
    summary = QwenPresentationSummary.model_validate_json(
        (root / "summary.json").read_text(encoding="utf-8")
    )
    if summary.case_count != len(records):
        raise ValueError("new Qwen records differ from summary")
    return records, summary


def _validate_media(
    old: _OldQwenRecordProjection,
    new: QwenPresentationRecord,
) -> None:
    if (
        old.sample_id != new.sample_id
        or old.clip_uid != new.clip_uid
        or old.target_video_sha256 != new.target_video_sha256
    ):
        raise ValueError("old/new Qwen case provenance differs")
    old_pictures = [] if old.reference_contract is None else [
        (item.image_index, item.image_sha256) for item in old.reference_contract.pictures
    ]
    new_pictures = [] if new.reference_contract is None else [
        (item.image_index, item.image_sha256) for item in new.reference_contract.pictures
    ]
    if old_pictures != new_pictures:
        raise ValueError("old/new Qwen reference image provenance differs")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.symlink_to(source.resolve(strict=True))
    except OSError:
        shutil.copy2(source, destination)


def _materialize_media(
    root: Path,
    records: list[QwenPresentationRecord],
) -> dict[str, dict[str, object]]:
    output = {}
    for record in records:
        key = _sha256_text(record.sample_id)
        target = Path(record.target_video_path).expanduser().resolve(strict=True)
        if _sha256_file(target) != record.target_video_sha256:
            raise ValueError("comparison target video differs from recorded SHA")
        target_name = f"target-{record.target_video_sha256[:12]}{target.suffix or '.mp4'}"
        _link_or_copy(target, root / "media" / key / target_name)
        pictures = {}
        if record.reference_contract is not None:
            for picture in record.reference_contract.pictures:
                source = Path(picture.image_path).expanduser().resolve(strict=True)
                if _sha256_file(source) != picture.image_sha256:
                    raise ValueError("comparison Picture differs from recorded SHA")
                name = (
                    f"picture-{picture.image_index}-{picture.image_sha256[:12]}"
                    f"{source.suffix or '.png'}"
                )
                _link_or_copy(source, root / "media" / key / name)
                pictures[picture.image_index] = f"media/{key}/{name}"
        output[record.sample_id] = {
            "target": f"media/{key}/{target_name}",
            "pictures": pictures,
        }
    return output


def _new_summary_payload(summary: QwenPresentationSummary) -> dict[str, object]:
    return {
        "ready": summary.ready_count,
        "failed": summary.failed_count,
        "first_pass_ready": summary.first_pass_ready_count,
        "repaired_ready": summary.repaired_ready_count,
        "presentation_counts": summary.presentation_counts,
        "visible_binding_removed": summary.visible_entity_binding_removed_count,
        "visible_binding_added": summary.visible_entity_binding_added_count,
        "visible_binding_changed": summary.visible_entity_binding_changed_count,
        "uncertain": summary.presentation_counts.get("uncertain", 0),
    }


def _review_html(
    *,
    comparisons: list[QwenPresentationABComparison],
    qwen35_new: list[QwenPresentationRecord],
    qwen38_new: list[QwenPresentationRecord],
    media: dict[str, dict[str, object]],
    qwen35_model: str,
    qwen38_model: str,
) -> str:
    q35 = {item.sample_id: item for item in qwen35_new}
    q38 = {item.sample_id: item for item in qwen38_new}
    cards = []
    for comparison in comparisons:
        left = q35[comparison.sample_id]
        right = q38[comparison.sample_id]
        case_media = media[comparison.sample_id]
        picture_paths = case_media["pictures"]
        pictures = "" if left.reference_contract is None else "".join(
            "<img loading='lazy' alt='"
            + html.escape(item.picture_label, quote=True)
            + "' src='"
            + html.escape(picture_paths[item.image_index], quote=True)
            + "'>"
            for item in left.reference_contract.pictures
        )
        left_decisions = {
            item.fact_id: item for item in left.speech_presentation_decisions
        }
        right_decisions = {
            item.fact_id: item for item in right.speech_presentation_decisions
        }
        rows = []
        facts = [] if left.original_audio_facts is None else left.original_audio_facts.speech
        for fact in facts:
            d35 = left_decisions.get(fact.fact_id)
            d38 = right_decisions.get(fact.fact_id)
            rows.append(
                "<tr><td>"
                + html.escape(fact.fact_id)
                + "</td><td>"
                + f"{fact.start_time:.3f}-{fact.end_time:.3f}"
                + "</td><td>"
                + html.escape(fact.speaker_id)
                + "</td><td>"
                + html.escape(str(fact.entity_id))
                + "</td><td>"
                + html.escape("-" if d35 is None else d35.speech_presentation)
                + "</td><td>"
                + html.escape("-" if d35 is None else str(d35.visible_entity_id))
                + "</td><td>"
                + html.escape("-" if d38 is None else d38.speech_presentation)
                + "</td><td>"
                + html.escape("-" if d38 is None else str(d38.visible_entity_id))
                + "</td></tr>"
            )
        prompts = (
            (qwen35_model + " OLD", comparison.qwen35_old_prompt),
            (qwen35_model + " NEW", comparison.qwen35_new_prompt),
            (qwen38_model + " OLD", comparison.qwen38_old_prompt),
            (qwen38_model + " NEW", comparison.qwen38_new_prompt),
        )
        prompt_panels = "".join(
            "<section><h3>"
            + html.escape(label)
            + "</h3><pre>"
            + html.escape(prompt or "[unavailable]")
            + "</pre></section>"
            for label, prompt in prompts
        )
        cards.append(
            "<article><h2>"
            + html.escape(comparison.sample_id)
            + "</h2><video controls preload='metadata' src='"
            + html.escape(str(case_media["target"]), quote=True)
            + "'></video><div class='pictures'>"
            + pictures
            + "</div><table><thead><tr><th>fact</th><th>time</th><th>speaker</th>"
            + "<th>old Final entity</th><th>Qwen3.5 NEW presentation</th>"
            + "<th>Qwen3.5 NEW entity</th><th>Qwen3.8 NEW presentation</th>"
            + "<th>Qwen3.8 NEW entity</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table><div class='prompts'>"
            + prompt_panels
            + "</div></article>"
        )
    return """<!doctype html><html><head><meta charset='utf-8'><title>Qwen speech presentation A/B</title>
<style>body{font:14px system-ui;margin:24px;background:#f4f4f2;color:#181818}article{background:#fff;border:1px solid #bbb;padding:18px;margin-bottom:24px}video{width:min(900px,100%);max-height:520px;background:#111}.pictures{display:flex;gap:8px;overflow:auto;margin:12px 0}.pictures img{width:180px;height:145px;object-fit:contain;background:#eee}table{width:100%;border-collapse:collapse}th,td{border:1px solid #ccc;padding:6px;text-align:left}.prompts{display:grid;grid-template-columns:1fr 1fr;gap:12px}.prompts section{min-width:0}.prompts pre{white-space:pre-wrap;background:#f3f3f3;padding:10px;max-height:520px;overflow:auto}@media(max-width:900px){.prompts{grid-template-columns:1fr}}</style></head><body><h1>Qwen speech-presentation old/new review</h1>""" + "".join(cards) + "</body></html>"


def build_qwen_speech_presentation_ab_review(
    *,
    qwen35_old_root: Path,
    qwen35_new_root: Path,
    qwen38_old_root: Path,
    qwen38_new_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> QwenPresentationABSummary:
    roots = {
        "qwen35_old": qwen35_old_root.expanduser().resolve(strict=True),
        "qwen35_new": qwen35_new_root.expanduser().resolve(strict=True),
        "qwen38_old": qwen38_old_root.expanduser().resolve(strict=True),
        "qwen38_new": qwen38_new_root.expanduser().resolve(strict=True),
    }
    manifests = {name: _manifest(root) for name, root in roots.items()}
    canonical_cases = manifests["qwen35_old"][0]
    canonical_payload = [item.model_dump(mode="json") for item in canonical_cases]
    if any(
        [item.model_dump(mode="json") for item in cases] != canonical_payload
        for cases, _ in manifests.values()
    ):
        raise ValueError("four-way Qwen comparison manifests differ")
    q35_old, q35_old_model = _old_output(roots["qwen35_old"])
    q38_old, q38_old_model = _old_output(roots["qwen38_old"])
    q35_new, q35_summary = _new_output(roots["qwen35_new"])
    q38_new, q38_summary = _new_output(roots["qwen38_new"])
    if q35_summary.source_h3_samples_sha256 != q38_summary.source_h3_samples_sha256:
        raise ValueError("Qwen3.5/Qwen3.8 presentation sources differ")
    expected_ids = [item.sample_id for item in canonical_cases]
    groups = (q35_old, q35_new, q38_old, q38_new)
    if any([item.sample_id for item in group] != expected_ids for group in groups):
        raise ValueError("four-way Qwen record order differs from case manifest")
    comparisons = []
    for old35, new35, old38, new38 in zip(*groups, strict=True):
        _validate_media(old35, new35)
        _validate_media(old38, new38)
        if (
            new35.clip_uid != new38.clip_uid
            or new35.target_video_sha256 != new38.target_video_sha256
            or new35.reference_contract != new38.reference_contract
            or new35.original_audio_facts != new38.original_audio_facts
        ):
            raise ValueError("Qwen3.5/Qwen3.8 presentation case provenance differs")
        comparisons.append(
            QwenPresentationABComparison(
                sample_id=new35.sample_id,
                clip_uid=new35.clip_uid,
                speech_decisions={
                    "qwen35": [
                        item.model_dump(mode="json")
                        for item in new35.speech_presentation_decisions
                    ],
                    "qwen38": [
                        item.model_dump(mode="json")
                        for item in new38.speech_presentation_decisions
                    ],
                },
                qwen35_old_prompt=old35.rendered_h3_prompt,
                qwen35_new_prompt=new35.rendered_h3_prompt,
                qwen38_old_prompt=old38.rendered_h3_prompt,
                qwen38_new_prompt=new38.rendered_h3_prompt,
            )
        )
    summary = QwenPresentationABSummary(
        case_count=len(comparisons),
        manifest_sha256_by_root={
            name: digest for name, (_, digest) in manifests.items()
        },
        qwen35_model_name=q35_summary.served_model_name,
        qwen38_model_name=q38_summary.served_model_name,
        qwen35=_new_summary_payload(q35_summary),
        qwen38=_new_summary_payload(q38_summary),
    )
    if q35_old_model != q35_summary.served_model_name:
        raise ValueError("Qwen3.5 old/new served model names differ")
    if q38_old_model != q38_summary.served_model_name:
        raise ValueError("Qwen3.8 old/new served model names differ")
    destination = output_root.expanduser().resolve(strict=False)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Qwen A/B review output already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.mkdir()
        media = _materialize_media(temporary, q35_new)
        (temporary / "comparisons.jsonl").write_text(
            "".join(
                _compact_json(item.model_dump(mode="json")) + "\n"
                for item in comparisons
            ),
            encoding="utf-8",
        )
        (temporary / "summary.json").write_text(
            json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
        (temporary / "review.html").write_text(
            _review_html(
                comparisons=comparisons,
                qwen35_new=q35_new,
                qwen38_new=q38_new,
                media=media,
                qwen35_model=q35_summary.served_model_name,
                qwen38_model=q38_summary.served_model_name,
            ),
            encoding="utf-8",
        )
        if destination.exists():
            backup = destination.with_name(
                f".{destination.name}.backup-{uuid.uuid4().hex}"
            )
            destination.replace(backup)
            try:
                temporary.replace(destination)
            except Exception:
                backup.replace(destination)
                raise
            shutil.rmtree(backup)
        else:
            temporary.replace(destination)
        return summary
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


__all__ = [
    "QWEN_PRESENTATION_AB_COMPARISON_VERSION",
    "QWEN_PRESENTATION_AB_SUMMARY_VERSION",
    "QwenPresentationABComparison",
    "QwenPresentationABSummary",
    "build_qwen_speech_presentation_ab_review",
]
