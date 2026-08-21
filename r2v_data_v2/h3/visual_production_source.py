from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from pydantic import model_validator

from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.v3.production_export import ProductionReference, ProductionSample
from r2v_data_v2.v3.schemas import ClipRecord


class ReadableClipIdentity(SchemaModel):
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str

    @model_validator(mode="after")
    def validate_identity(self) -> ReadableClipIdentity:
        for name in (
            "clip_display_path",
            "media_collection_relpath",
            "media_collection_name",
            "episode_name",
            "clip_name",
            "shard_id",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"readable clip identity {name} is empty")
        return self


class VisualProductionClip(SchemaModel):
    identity: ReadableClipIdentity
    sample: ProductionSample
    clip: ClipRecord
    clip_record_path: str
    subject_references: list[ProductionReference]

    @model_validator(mode="after")
    def validate_clip(self) -> VisualProductionClip:
        if self.sample.clip_uid != self.identity.clip_uid:
            raise ValueError("canonical sample and readable identity differ")
        if self.clip.clip_uid != self.identity.clip_uid:
            raise ValueError("canonical sample and clip record differ")
        if any(reference.kind != "subject" for reference in self.subject_references):
            raise ValueError("subject_references must contain only subjects")
        return self


class VisualProductionInventory(SchemaModel):
    visual_production_root: str
    visual_runs_root: str
    canonical_sample_count: int
    eligible_clip_count: int
    eligible_subject_occurrence_count: int
    media_collection_count: int
    media_collection_clip_counts: dict[str, int]
    shard_count: int
    clips: list[VisualProductionClip]
    skip_reason_counts: dict[str, int]


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid canonical samples.jsonl line {line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(
                    f"canonical samples.jsonl line {line_number} is not an object"
                )
            yield value


def _relative_posix_path(value: object, *, field_name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value.strip() or "\\" in value:
        raise ValueError(f"{field_name} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{field_name} must remain inside the source collection")
    return path


def derive_readable_clip_identity(
    *,
    clip_uid: str,
    shard_id: str,
    source_relative_video_path: object,
    source_relative_source_video_path: object,
) -> ReadableClipIdentity:
    video_path = _relative_posix_path(
        source_relative_video_path,
        field_name="source_relative_video_path",
    )
    source_video_path = _relative_posix_path(
        source_relative_source_video_path,
        field_name="source_relative_source_video_path",
    )
    media_collection = source_video_path.parent
    if str(media_collection) in {"", "."}:
        raise ValueError("source video must belong to a media collection directory")
    return ReadableClipIdentity(
        clip_uid=clip_uid,
        clip_display_path=video_path.with_suffix("").as_posix(),
        media_collection_relpath=media_collection.as_posix(),
        media_collection_name=media_collection.name,
        episode_name=source_video_path.stem,
        clip_name=video_path.stem,
        shard_id=shard_id,
    )


def readable_output_path(
    root: Path, identity: ReadableClipIdentity, suffix: str
) -> Path:
    if not suffix.startswith("."):
        raise ValueError("readable output suffix must start with a dot")
    relative = PurePosixPath(identity.clip_display_path)
    return root.joinpath(*relative.parts).with_suffix(suffix)


def load_visual_production_inventory(
    *,
    visual_production_root: Path,
    visual_runs_root: Path,
) -> VisualProductionInventory:
    production_root = visual_production_root.expanduser().resolve(strict=True)
    runs_root = visual_runs_root.expanduser().resolve(strict=True)
    samples_path = production_root / "samples.jsonl"
    if not samples_path.is_file():
        raise FileNotFoundError("canonical Visual Production samples.jsonl is missing")

    clips: list[VisualProductionClip] = []
    sample_ids: set[str] = set()
    clip_uids: set[str] = set()
    skipped: Counter[str] = Counter()
    canonical_count = 0
    for raw in _read_jsonl(samples_path):
        sample = ProductionSample.model_validate(raw)
        canonical_count += 1
        if sample.sample_id in sample_ids:
            raise ValueError(f"duplicate canonical sample_id: {sample.sample_id}")
        if sample.clip_uid in clip_uids:
            raise ValueError(f"duplicate canonical clip_uid: {sample.clip_uid}")
        sample_ids.add(sample.sample_id)
        clip_uids.add(sample.clip_uid)

        clip_path = (
            runs_root / sample.source.shard_id / "clips" / sample.clip_uid / "clip.json"
        ).resolve(strict=True)
        clip_path.relative_to(runs_root)
        clip = ClipRecord.model_validate_json(clip_path.read_text(encoding="utf-8"))
        if clip.clip_uid != sample.clip_uid:
            raise ValueError("canonical sample does not match its clip.json")
        if sample.target_video != clip.source.video_path:
            raise ValueError(
                "canonical target_video differs from clip source video_path"
            )
        metadata = clip.source.metadata
        identity = derive_readable_clip_identity(
            clip_uid=sample.clip_uid,
            shard_id=sample.source.shard_id,
            source_relative_video_path=metadata.get("source_relative_video_path"),
            source_relative_source_video_path=metadata.get(
                "source_relative_source_video_path"
            ),
        )
        subject_references = [
            reference for reference in sample.references if reference.kind == "subject"
        ]
        for reference in subject_references:
            reference_path = (production_root / reference.image_path).resolve(
                strict=True
            )
            reference_path.relative_to(production_root)
            if not reference_path.is_file():
                raise ValueError("canonical subject reference is not a file")
        if not subject_references:
            skipped.update(["no_subject_reference"])
            continue
        clips.append(
            VisualProductionClip(
                identity=identity,
                sample=sample,
                clip=clip,
                clip_record_path=str(clip_path),
                subject_references=subject_references,
            )
        )

    collection_counts = Counter(
        item.identity.media_collection_relpath for item in clips
    )
    return VisualProductionInventory(
        visual_production_root=str(production_root),
        visual_runs_root=str(runs_root),
        canonical_sample_count=canonical_count,
        eligible_clip_count=len(clips),
        eligible_subject_occurrence_count=sum(
            len(item.subject_references) for item in clips
        ),
        media_collection_count=len(collection_counts),
        media_collection_clip_counts=dict(sorted(collection_counts.items())),
        shard_count=len({item.identity.shard_id for item in clips}),
        clips=clips,
        skip_reason_counts=dict(sorted(skipped.items())),
    )
