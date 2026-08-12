from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median

from PIL import Image, ImageDraw

from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.schemas import ClipRecord, ReferenceType, TrackedMasksArtifact

_COMPOSITION_KEYS = (
    "subject_only",
    "object_only",
    "subject_object",
    "group_only",
    "group_subject",
    "group_object",
    "other",
)
_ENTITY_TYPES: tuple[ReferenceType, ...] = ("subject", "object", "group")


def _composition(types: set[str]) -> str:
    return {
        frozenset({"subject"}): "subject_only",
        frozenset({"object"}): "object_only",
        frozenset({"subject", "object"}): "subject_object",
        frozenset({"group"}): "group_only",
        frozenset({"group", "subject"}): "group_subject",
        frozenset({"group", "object"}): "group_object",
    }.get(frozenset(types), "other")


def _iter_clips(run_root: Path) -> list[ClipRecord]:
    return [
        ClipRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((run_root / "clips").glob("*/clip.json"))
    ]


def _read_masks(run_root: Path, clip_uid: str) -> TrackedMasksArtifact | None:
    path = run_root / "clips" / clip_uid / "masks.rle.json"
    if not path.is_file():
        return None
    return TrackedMasksArtifact.model_validate_json(path.read_text(encoding="utf-8"))


def _resolved_reference_path(run_root: Path, image_path: str) -> Path:
    value = Path(image_path).expanduser()
    candidate = value if value.is_absolute() else run_root / value
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(run_root)
    except ValueError as exc:
        raise ValueError("reference image escapes the audited run root") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"reference image is missing: {resolved}")
    return resolved


def _write_contact_sheet(
    run_root: Path,
    output_root: Path,
    clip: ClipRecord,
    category: str,
    entity_ids: list[str],
) -> None:
    assert clip.pairing is not None
    references = {
        item.entity_id: item
        for item in clip.references.entities
        if item.status == "ready" and item.image_path is not None
    }
    source_images: list[tuple[str, Image.Image]] = []
    entities_by_id = {
        item.entity_id: item
        for item in (clip.annotation.entities if clip.annotation is not None else [])
    }
    for entity_id in entity_ids:
        reference = references.get(entity_id)
        if reference is None or reference.image_path is None:
            continue
        path = _resolved_reference_path(run_root, reference.image_path)
        with Image.open(path) as opened:
            entity = entities_by_id.get(entity_id)
            if entity is None:
                continue
            label = " | ".join(
                (
                    entity_id,
                    entity.reference_type,
                    reference.reference_scope,
                    "synthetic" if reference.synthetic else "real",
                    entity.phrase,
                )
            )
            source_images.append((label, opened.convert("RGB")))
    if not source_images:
        return

    tile_width, tile_height, title_height = 360, 340, 44
    sheet = Image.new(
        "RGB",
        (tile_width * len(source_images), tile_height + title_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"{clip.clip_uid} | {category}", fill="black")
    for index, (label, source) in enumerate(source_images):
        image = source.copy()
        image.thumbnail((tile_width, tile_height - 24), Image.Resampling.LANCZOS)
        x = index * tile_width + (tile_width - image.width) // 2
        y = title_height + (tile_height - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text(
            (index * tile_width + 8, title_height + 6),
            label,
            fill="black",
        )
    destination = output_root / "contact_sheets" / category / f"{clip.clip_uid}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="JPEG", quality=92)


def _write_integrity_sheet(
    run_root: Path,
    output_root: Path,
    clip: ClipRecord,
    *,
    category: str,
    entity_id: str,
) -> None:
    if clip.reference_integrity is None:
        return
    state = next(
        (item for item in clip.reference_integrity.entities if item.entity_id == entity_id),
        None,
    )
    if state is None or state.source_context_path is None:
        return
    entity = next(
        (
            item
            for item in (clip.annotation.entities if clip.annotation is not None else [])
            if item.entity_id == entity_id
        ),
        None,
    )
    if entity is None:
        return
    source_path = _resolved_reference_path(run_root, state.source_context_path)
    final_path = _resolved_reference_path(run_root, state.final_reference_path)
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    with Image.open(final_path) as opened:
        final = opened.convert("RGB")
    panel_width, panel_height, title_height = 420, 420, 86
    sheet = Image.new("RGB", (panel_width * 2, panel_height + title_height), "white")
    draw = ImageDraw.Draw(sheet)
    label = " | ".join(
        (
            clip.clip_uid,
            entity_id,
            entity.reference_type,
            entity.phrase,
            state.input_reference.reference_scope,
            "synthetic" if state.input_reference.synthetic else "real",
            state.status,
            state.reason,
        )
    )
    draw.text((8, 8), label, fill="black")
    for index, (heading, image) in enumerate(
        (("SOURCE CONTEXT", source), ("FINAL REFERENCE", final))
    ):
        image.thumbnail((panel_width - 20, panel_height - 30), Image.Resampling.LANCZOS)
        x = index * panel_width + (panel_width - image.width) // 2
        y = title_height + (panel_height - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((index * panel_width + 8, title_height + 4), heading, fill="black")
    destination = (
        output_root
        / "contact_sheets"
        / category
        / f"{clip.clip_uid}_{entity_id}.jpg"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="JPEG", quality=92)


def _count_distribution(
    values: list[int], *, maximum: int | None = None
) -> dict[str, object]:
    counts = Counter(values)
    if maximum is None:
        maximum = max(values, default=0)
    return {
        "mean": mean(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "histogram": {str(value): counts[value] for value in range(maximum + 1)},
    }


def audit_entity_composition(
    *,
    run_root: Path,
    output_root: Path,
    contact_sheets: bool = False,
) -> dict[str, object]:
    source_root = run_root.expanduser().resolve(strict=True)
    destination = output_root.expanduser().resolve(strict=False)
    if destination == source_root or source_root in destination.parents:
        raise ValueError("composition audit output_root must be outside run_root")

    composition_counts: Counter[str] = Counter()
    final_entity_counts: Counter[str] = Counter()
    funnel = {
        stage: Counter[str]()
        for stage in (
            "annotated",
            "segment_ready",
            "segment_not_found",
            "segment_failed",
            "coverage_qualified",
            "pair_retained",
            "reference_edit_ready",
            "reference_integrity_ready",
            "export_ready",
            "final_ready",
        )
    }
    mixed_cases: list[dict[str, object]] = []
    sample_count = samples_with_subject = samples_with_object = 0
    samples_with_both = 0
    annotation_ready_clip_count = 0
    annotation_entity_counts: Counter[str] = Counter()
    annotation_density: list[int] = []
    final_density: list[int] = []
    stage_density: dict[str, list[int]] = {
        "post_pair": [],
        "post_reference_edit": [],
        "post_reference_integrity": [],
        "final_export": final_density,
    }
    integrity_rejection_reasons: Counter[str] = Counter()
    final_synthetic_counts: Counter[str] = Counter()
    final_scope_counts: Counter[str] = Counter()
    background_reference_count = 0

    for clip in _iter_clips(source_root):
        annotation = clip.annotation
        if annotation is None or annotation.status != "ready":
            continue
        annotation_ready_clip_count += 1
        annotation_density.append(len(annotation.entities))
        entities_by_id = {entity.entity_id: entity for entity in annotation.entities}
        for entity in annotation.entities:
            funnel["annotated"][entity.reference_type] += 1
            annotation_entity_counts[entity.reference_type] += 1

        masks = _read_masks(source_root, clip.clip_uid)
        if masks is not None:
            for entity_id, track in masks.entities.items():
                entity = entities_by_id.get(entity_id)
                if entity is None:
                    continue
                funnel[f"segment_{track.status}"][entity.reference_type] += 1

        if clip.coverage is not None:
            for entity_id in clip.coverage.qualifying_entity_ids:
                entity = entities_by_id.get(entity_id)
                if entity is not None:
                    funnel["coverage_qualified"][entity.reference_type] += 1

        post_integrity_ids = (
            list(clip.pairing.retained_entity_ids)
            if clip.pairing is not None and clip.pairing.status == "ready"
            else []
        )
        post_reference_edit_ids = post_integrity_ids
        post_pair_ids = post_reference_edit_ids
        has_pair_evidence = clip.pairing is not None
        if (
            clip.reference_integrity is not None
            and clip.reference_integrity.status == "ready"
        ):
            post_reference_edit_ids = [
                item.entity_id for item in clip.reference_integrity.entities
            ]
            post_pair_ids = post_reference_edit_ids
            has_pair_evidence = True
        if clip.reference_edit is not None and clip.reference_edit.status == "ready":
            post_pair_ids = [item.entity_id for item in clip.reference_edit.entities]
            has_pair_evidence = True
        if has_pair_evidence:
            for entity_id in post_pair_ids:
                entity = entities_by_id.get(entity_id)
                if entity is not None:
                    funnel["pair_retained"][entity.reference_type] += 1
            for entity_id in post_reference_edit_ids:
                entity = entities_by_id.get(entity_id)
                if entity is not None:
                    funnel["reference_edit_ready"][entity.reference_type] += 1
            for entity_id in post_integrity_ids:
                entity = entities_by_id.get(entity_id)
                if entity is not None:
                    funnel["reference_integrity_ready"][entity.reference_type] += 1
                    funnel["final_ready"][entity.reference_type] += 1
            stage_density["post_pair"].append(len(post_pair_ids))
            stage_density["post_reference_edit"].append(len(post_reference_edit_ids))
            stage_density["post_reference_integrity"].append(len(post_integrity_ids))

        if (
            clip.reference_integrity is not None
            and clip.reference_integrity.status == "ready"
        ):
            for item in clip.reference_integrity.entities:
                if item.status == "rejected":
                    integrity_rejection_reasons[item.reason] += 1
                    if contact_sheets:
                        _write_integrity_sheet(
                            source_root,
                            destination,
                            clip,
                            category="integrity_rejected",
                            entity_id=item.entity_id,
                        )
                elif item.diagnostics.suspicious and item.status == "accepted":
                    if contact_sheets:
                        _write_integrity_sheet(
                            source_root,
                            destination,
                            clip,
                            category="integrity_suspicious_accepted",
                            entity_id=item.entity_id,
                        )

        if (
            not clip.export.accepted
            or clip.pairing is None
            or clip.pairing.status != "ready"
        ):
            continue
        retained = [
            entity_id
            for entity_id in clip.pairing.retained_entity_ids
            if entity_id in entities_by_id
        ]
        types = {entities_by_id[entity_id].reference_type for entity_id in retained}
        category = _composition(types)
        composition_counts[category] += 1
        sample_count += 1
        final_density.append(len(retained))
        samples_with_subject += int("subject" in types)
        samples_with_object += int("object" in types)
        samples_with_both += int({"subject", "object"}.issubset(types))
        for entity_id in retained:
            reference_type = entities_by_id[entity_id].reference_type
            final_entity_counts[reference_type] += 1
            funnel["export_ready"][reference_type] += 1
            reference = next(
                item for item in clip.references.entities if item.entity_id == entity_id
            )
            final_synthetic_counts["synthetic" if reference.synthetic else "real"] += 1
            final_scope_counts[reference.reference_scope] += 1
        if (
            clip.pairing.background_token is not None
            and clip.references.background is not None
            and clip.references.background.status in {"clean_raw", "ready_removed"}
        ):
            background_reference_count += 1
        if {"subject", "object"}.issubset(types):
            mixed_cases.append(
                {
                    "clip_uid": clip.clip_uid,
                    "parent_video_id": clip.source.parent_video_id,
                    "retained_entity_ids": retained,
                    "reference_types": [
                        entities_by_id[entity_id].reference_type
                        for entity_id in retained
                    ],
                }
            )
        if contact_sheets:
            subject_ids = [
                entity_id
                for entity_id in retained
                if entities_by_id[entity_id].reference_type == "subject"
            ]
            object_ids = [
                entity_id
                for entity_id in retained
                if entities_by_id[entity_id].reference_type == "object"
            ]
            for sheet_category, entity_ids in (
                ("subjects", subject_ids),
                ("objects", object_ids),
                ("all_entities", retained),
                (
                    "multi_entity_samples",
                    retained if len(retained) > 1 else [],
                ),
            ):
                if entity_ids:
                    _write_contact_sheet(
                        source_root,
                        destination,
                        clip,
                        sheet_category,
                        entity_ids,
                    )

    summary: dict[str, object] = {
        "annotation_ready_clip_count": annotation_ready_clip_count,
        "annotation_entity_count": {
            entity_type: annotation_entity_counts[entity_type]
            for entity_type in _ENTITY_TYPES
        },
        "annotation_entities_per_ready_clip": _count_distribution(
            annotation_density,
            maximum=8,
        ),
        "final_accepted_sample_count": sample_count,
        "final_entity_reference_count": {
            entity_type: final_entity_counts[entity_type]
            for entity_type in _ENTITY_TYPES
        },
        "final_entity_references_per_accepted_sample": _count_distribution(
            final_density,
            maximum=8,
        ),
        "reference_density_by_stage": {
            stage: _count_distribution(values, maximum=8)
            for stage, values in stage_density.items()
        },
        "integrity_rejections": {
            "count": sum(integrity_rejection_reasons.values()),
            "reasons": dict(sorted(integrity_rejection_reasons.items())),
        },
        "background_reference_count": background_reference_count,
        "final_entity_references_by_source": {
            "real": final_synthetic_counts["real"],
            "synthetic": final_synthetic_counts["synthetic"],
        },
        "final_entity_references_by_scope": {
            "full": final_scope_counts["full"],
            "local": final_scope_counts["local"],
        },
        "sample_count": sample_count,
        "compositions": {key: composition_counts[key] for key in _COMPOSITION_KEYS},
        "samples_with_subject": samples_with_subject,
        "samples_with_object": samples_with_object,
        "samples_with_subject_and_object": samples_with_both,
        "entity_counts": {
            entity_type: final_entity_counts[entity_type]
            for entity_type in _ENTITY_TYPES
        },
        "funnel_by_type": {
            stage: {entity_type: counts[entity_type] for entity_type in _ENTITY_TYPES}
            for stage, counts in funnel.items()
        },
        "mixed_subject_object_clip_count": len(mixed_cases),
    }
    destination.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination / "summary.json", summary)
    write_json_atomic(
        destination / "subject_object_clips.json",
        {"clips": mixed_cases},
    )
    csv_path = destination / "subject_object_clips.csv"
    temporary = csv_path.with_suffix(".csv.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "clip_uid",
                    "parent_video_id",
                    "retained_entity_ids",
                    "reference_types",
                ),
            )
            writer.writeheader()
            for case in mixed_cases:
                writer.writerow(
                    {
                        **case,
                        "retained_entity_ids": ",".join(
                            str(value) for value in case["retained_entity_ids"]
                        ),
                        "reference_types": ",".join(
                            str(value) for value in case["reference_types"]
                        ),
                    }
                )
        temporary.replace(csv_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return summary
