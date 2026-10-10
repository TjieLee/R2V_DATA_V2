"""Read published V3 groups without inspecting or copying source media."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path

from r2v_data_v2.h3.schemas import entity_reference_order
from r2v_data_v2.h3.t2va_production import atomic_json
from r2v_data_v2.v3.production_export import ProductionSample

GROUP_NAME = re.compile(r"group-[0-9]{6}")


@dataclass(frozen=True)
class PublishedGroup:
    group_id: str
    directory: Path
    samples_path: Path
    declared_count: int


@dataclass(frozen=True)
class GroupTask:
    ordinal: int
    clip_uid: str
    sample_id: str
    sample: dict
    target_video_path: str
    pictures: list[dict]
    subjects: list[dict]


def discover_published_groups(source_root: Path) -> list[PublishedGroup]:
    if not source_root.exists():
        return []
    groups = []
    for directory in sorted(source_root.iterdir()):
        if not GROUP_NAME.fullmatch(directory.name):
            continue
        marker = source_root.parent / "state/resource_epochs" / directory.name / "completed.json"
        samples = directory / "samples.jsonl"
        if not marker.is_file() or not samples.is_file():
            continue
        record = json.loads(marker.read_text())
        count = record.get("sample_count")
        if (record.get("group_id") == directory.name
                and record.get("status") == "completed"
                and record.get("export_completed") is True
                and type(count) is int and count >= 0):
            groups.append(PublishedGroup(directory.name, directory, samples, count))
    return groups


def adapt_production_sample(sample: ProductionSample, group_directory: Path, ordinal: int) -> GroupTask:
    pictures = []
    for reference in sample.references:
        picture = reference.model_dump(mode="json")
        picture["picture_label"] = f"<Picture {reference.image_index}>"
        picture["image_path"] = str(group_directory / reference.image_path)
        pictures.append(picture)
    subjects = []

    def add(kind, labels, **ownership):
        index = len(subjects) + 1
        subjects.append(dict(subject_index=index, subject_label=f"<Subject {index}>",
                             kind=kind, source_picture_labels=labels, **ownership))

    for entity in entity_reference_order((p["kind"], p["entity_id"]) for p in pictures):
        add("entity", [p["picture_label"] for p in pictures if p["entity_id"] == entity], entity_id=entity)
    for picture in pictures:
        if picture["kind"] == "attribute":
            add("attribute", [picture["picture_label"]], **{
                k: picture[k] for k in ("attribute_id", "owner_entity_id", "attribute_type")
            })
    backgrounds = [p["picture_label"] for p in pictures if p["kind"] == "background"]
    if backgrounds:
        add("background", backgrounds)
    return GroupTask(ordinal, sample.clip_uid, sample.sample_id,
                     sample.model_dump(mode="json"), sample.target_video, pictures, subjects)


def freeze_pilot_inventory(group: PublishedGroup, destination: Path, limit: int, *, start_row=0) -> dict:
    if limit is None:
        raise ValueError("pilot requires a bounded limit")
    return freeze_group_inventory(group, destination, limit, start_row=start_row)


def freeze_group_inventory(group: PublishedGroup, destination: Path, limit: int | None, *, start_row=0,
                           group_parts=1) -> dict:
    if group_parts not in (1, 4) or (limit is not None and group_parts != 1):
        raise ValueError("group parts require full production scope and must be 1 or 4")
    if limit is None:
        if start_row:
            raise ValueError("full group cannot start partway through the source")
    elif not 1 <= limit <= 200:
        raise ValueError("CPU pilot limit must be between 1 and 200")
    if start_row < 0:
        raise ValueError("CPU pilot limit must be between 1 and 200 and start_row nonnegative")
    if destination.exists():
        metadata = json.loads((destination / "inventory.json").read_text())
        if (metadata["limit"] != limit or metadata["group_id"] != group.group_id
                or metadata.get("start_row", 0) != start_row
                or metadata.get("group_parts", 1) != group_parts):
            raise ValueError("resume inventory settings differ")
        return metadata
    if start_row and start_row >= group.declared_count:
        raise ValueError("selected source range is empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".inventory-", dir=destination.parent))
    try:
        offsets, seen, source_row = [], set(), 0
        with group.samples_path.open() as source, (temporary / "tasks.jsonl").open("wb") as output:
            for line in source:
                if not line.strip():
                    continue
                source_row += 1
                if source_row <= start_row:
                    continue
                sample = ProductionSample.model_validate_json(line)
                if sample.clip_uid in seen:
                    raise ValueError(f"duplicate clip identity: {sample.clip_uid}")
                seen.add(sample.clip_uid)
                task = adapt_production_sample(sample, group.directory, len(offsets))
                offsets.append(output.tell())
                output.write((json.dumps(asdict(task), ensure_ascii=False) + "\n").encode())
                if len(offsets) == limit:
                    break
            output.flush()
            os.fsync(output.fileno())
        expected = group.declared_count if limit is None else min(limit, group.declared_count - start_row)
        if len(offsets) != expected:
            raise ValueError("published source has fewer rows than its Export count")
        metadata = {"mode": "full_group" if limit is None else "pilot", "group_id": group.group_id, "source_directory": str(group.directory),
                        "source_samples_path": str(group.samples_path), "limit": limit, "start_row": start_row,
                        "selected_count": len(offsets), "declared_source_count": group.declared_count,
                        "offsets": offsets}
        if group_parts > 1:
            count = len(offsets)
            boundaries = [count * part // group_parts for part in range(group_parts + 1)]
            metadata.update(group_parts=group_parts, parts=[
                {"start_ordinal": start, "end_ordinal": end}
                for start, end in pairwise(boundaries) if start < end]
                or [{"start_ordinal": 0, "end_ordinal": 0}])
        atomic_json(temporary / "inventory.json", metadata)
        temporary.rename(destination)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_inventory_task(inventory_root: Path, ordinal: int, *, metadata=None) -> GroupTask:
    if metadata is None:
        metadata = json.loads((inventory_root / "inventory.json").read_text())
    with (inventory_root / "tasks.jsonl").open("rb") as source:
        source.seek(metadata["offsets"][ordinal])
        return GroupTask(**json.loads(source.readline()))
