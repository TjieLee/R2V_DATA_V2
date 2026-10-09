"""Read published V3 groups without inspecting or copying source media."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
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


def freeze_pilot_inventory(group: PublishedGroup, destination: Path, limit: int) -> dict:
    if not 1 <= limit <= 200:
        raise ValueError("CPU pilot limit must be between 1 and 200")
    if destination.exists():
        metadata = json.loads((destination / "inventory.json").read_text())
        if metadata["limit"] != limit or metadata["group_id"] != group.group_id:
            raise ValueError("resume inventory settings differ")
        return metadata
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".inventory-", dir=destination.parent))
    try:
        offsets, seen = [], set()
        with group.samples_path.open() as source, (temporary / "tasks.jsonl").open("wb") as output:
            for line in source:
                if not line.strip():
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
        if len(offsets) != min(limit, group.declared_count):
            raise ValueError("published source has fewer rows than its Export count")
        metadata = dict(mode="pilot", group_id=group.group_id, source_directory=str(group.directory),
                        source_samples_path=str(group.samples_path), limit=limit,
                        selected_count=len(offsets), declared_source_count=group.declared_count,
                        offsets=offsets)
        atomic_json(temporary / "inventory.json", metadata)
        temporary.rename(destination)
        return metadata
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_inventory_task(inventory_root: Path, ordinal: int) -> GroupTask:
    metadata = json.loads((inventory_root / "inventory.json").read_text())
    with (inventory_root / "tasks.jsonl").open("rb") as source:
        source.seek(metadata["offsets"][ordinal])
        return GroupTask(**json.loads(source.readline()))
