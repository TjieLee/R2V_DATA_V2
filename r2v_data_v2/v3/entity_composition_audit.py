from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

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
    composition: str,
) -> None:
    assert clip.pairing is not None
    references = {
        item.entity_id: item
        for item in clip.references.entities
        if item.status == "ready" and item.image_path is not None
    }
    source_images: list[tuple[str, Image.Image]] = []
    for entity_id in clip.pairing.retained_entity_ids:
        reference = references.get(entity_id)
        if reference is None or reference.image_path is None:
            continue
        path = _resolved_reference_path(run_root, reference.image_path)
        with Image.open(path) as opened:
            source_images.append((entity_id, opened.convert("RGB")))
    if not source_images:
        return

    tile_width, tile_height, title_height = 320, 320, 44
    sheet = Image.new(
        "RGB",
        (tile_width * len(source_images), tile_height + title_height),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"{clip.clip_uid} | {composition}", fill="black")
    for index, (entity_id, source) in enumerate(source_images):
        image = source.copy()
        image.thumbnail((tile_width, tile_height - 24), Image.Resampling.LANCZOS)
        x = index * tile_width + (tile_width - image.width) // 2
        y = title_height + (tile_height - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((index * tile_width + 8, title_height + 6), entity_id, fill="black")
    destination = output_root / "contact_sheets" / composition / f"{clip.clip_uid}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="JPEG", quality=92)


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
            "final_ready",
        )
    }
    mixed_cases: list[dict[str, object]] = []
    sample_count = samples_with_subject = samples_with_object = 0
    samples_with_both = 0

    for clip in _iter_clips(source_root):
        annotation = clip.annotation
        if annotation is None or annotation.status != "ready":
            continue
        entities_by_id = {entity.entity_id: entity for entity in annotation.entities}
        for entity in annotation.entities:
            funnel["annotated"][entity.reference_type] += 1

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

        if clip.pairing is not None and clip.pairing.status == "ready":
            for entity_id in clip.pairing.retained_entity_ids:
                entity = entities_by_id.get(entity_id)
                if entity is not None:
                    funnel["pair_retained"][entity.reference_type] += 1

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
        samples_with_subject += int("subject" in types)
        samples_with_object += int("object" in types)
        samples_with_both += int({"subject", "object"}.issubset(types))
        for entity_id in retained:
            reference_type = entities_by_id[entity_id].reference_type
            final_entity_counts[reference_type] += 1
            funnel["final_ready"][reference_type] += 1
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
        if contact_sheets and category in {
            "subject_only",
            "object_only",
            "subject_object",
        }:
            _write_contact_sheet(source_root, destination, clip, category)

    summary: dict[str, object] = {
        "sample_count": sample_count,
        "compositions": {
            key: composition_counts[key] for key in _COMPOSITION_KEYS
        },
        "samples_with_subject": samples_with_subject,
        "samples_with_object": samples_with_object,
        "samples_with_subject_and_object": samples_with_both,
        "entity_counts": {
            entity_type: final_entity_counts[entity_type]
            for entity_type in _ENTITY_TYPES
        },
        "funnel_by_type": {
            stage: {
                entity_type: counts[entity_type]
                for entity_type in _ENTITY_TYPES
            }
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
