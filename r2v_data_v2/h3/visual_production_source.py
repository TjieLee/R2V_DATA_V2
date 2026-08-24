from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from itertools import chain
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import field_validator, model_validator

from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.v3.production_export import (
    PRODUCTION_SAMPLE_SCHEMA_VERSION,
    ProductionReference,
    ProductionReferenceKind,
    ProductionSample,
)
from r2v_data_v2.v3.schemas import (
    SAMPLE_SCHEMA_VERSION,
    ClipRecord,
    DatasetReference,
    DatasetSample,
)
from r2v_data_v2.v3.subject_attributes import EnrichedSample

VisualInputSchema = Literal[
    "r2v.v3.production_sample.1",
    "r2v.v3.sample.1",
]
VisualInputMode = Literal["compacted_production", "single_run_export"]


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


class NormalizedVisualReference(ProductionReference):
    artifact_path: str

    @field_validator("artifact_path")
    @classmethod
    def validate_artifact_path(cls, value: str) -> str:
        if not value.strip() or not Path(value).is_absolute():
            raise ValueError("normalized reference artifact_path must be absolute")
        return value


class NormalizedVisualSample(SchemaModel):
    sample_id: str
    clip_uid: str
    target_video: str
    t2v_caption: str
    r2v_instruction: str
    references: list[NormalizedVisualReference]

    @model_validator(mode="after")
    def validate_references(self) -> NormalizedVisualSample:
        indexes = [reference.image_index for reference in self.references]
        if indexes != list(range(1, len(indexes) + 1)):
            raise ValueError("normalized reference indexes must be contiguous")
        return self


class VisualProductionClip(SchemaModel):
    identity: ReadableClipIdentity
    sample: NormalizedVisualSample
    clip: ClipRecord
    clip_record_path: str
    subject_references: list[NormalizedVisualReference]

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
    visual_input_schema: VisualInputSchema
    visual_input_mode: VisualInputMode
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


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    if not relative_path or relative_path.startswith("/"):
        raise ValueError("Visual artifact path must be relative")
    artifact = (root / relative_path).resolve(strict=True)
    artifact.relative_to(root)
    if not artifact.is_file():
        raise ValueError("Visual artifact is not a file")
    return artifact


def _dataset_reference_kind(reference: DatasetReference) -> ProductionReferenceKind:
    if reference.type == "background":
        return "background"
    for kind in ("subject", "object", "group"):
        if reference.token.startswith(f"<ref_{kind}_"):
            return kind
    raise ValueError(f"cannot derive Visual reference kind from {reference.token}")


def _normalize_production_reference(
    reference: ProductionReference,
    *,
    production_root: Path,
) -> NormalizedVisualReference:
    artifact = _resolve_artifact(production_root, reference.image_path)
    return NormalizedVisualReference(
        **reference.model_dump(mode="python"),
        artifact_path=str(artifact),
    )


def _normalize_dataset_reference(
    reference: DatasetReference,
    *,
    image_index: int,
    export_root: Path,
) -> NormalizedVisualReference:
    artifact = _resolve_artifact(export_root, reference.image_path)
    return NormalizedVisualReference(
        image_id=f"image_{image_index}",
        image_index=image_index,
        kind=_dataset_reference_kind(reference),
        entity_id=reference.entity_id,
        image_path=reference.image_path,
        artifact_path=str(artifact),
        source_frame_index=reference.source_frame_index,
        scope=reference.scope,
        visible_region=reference.visible_region,
        synthetic=reference.synthetic,
    )


def _load_enriched_samples(run_root: Path) -> dict[str, EnrichedSample]:
    path = run_root / "subject_attributes" / "enriched_samples.jsonl"
    if not path.is_file():
        return {}
    values: dict[str, EnrichedSample] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            enriched = EnrichedSample.model_validate_json(line)
            if Path(enriched.source_run_root).expanduser().resolve(strict=False) != run_root:
                raise ValueError(
                    f"enriched source_run_root mismatch at {path}:{line_number}"
                )
            if enriched.sample_id in values:
                raise ValueError(f"duplicate enriched sample_id: {enriched.sample_id}")
            values[enriched.sample_id] = enriched
    return values


def _normalize_enriched_sample(
    sample: DatasetSample,
    enriched: EnrichedSample,
    *,
    run_root: Path,
) -> NormalizedVisualSample:
    if enriched.clip_uid != sample.sample_id:
        raise ValueError(f"enriched clip_uid mismatch: {enriched.sample_id}")
    original_target = enriched.original_visual.get("target_video")
    if original_target is not None and original_target != sample.target_video:
        raise ValueError(f"enriched target_video mismatch: {enriched.sample_id}")
    original_source = enriched.original_visual.get("source")
    if original_source is not None and original_source != sample.source.model_dump():
        raise ValueError(f"enriched source provenance mismatch: {enriched.sample_id}")

    entity_references: dict[str, DatasetReference] = {}
    background_reference: DatasetReference | None = None
    for reference in sample.references:
        if reference.type == "background":
            if background_reference is not None:
                raise ValueError("Visual sample contains duplicate background references")
            background_reference = reference
        else:
            assert reference.entity_id is not None
            if reference.entity_id in entity_references:
                raise ValueError(
                    f"Visual sample contains duplicate entity_id: {reference.entity_id}"
                )
            entity_references[reference.entity_id] = reference

    attributes = {
        record.attribute_id: record for record in enriched.accepted_attributes
    }
    visual_entity_ids = set(entity_references)
    if len(attributes) != len(enriched.accepted_attributes):
        raise ValueError(f"duplicate accepted attribute_id: {enriched.sample_id}")
    normalized: list[NormalizedVisualReference] = []
    for enriched_reference in enriched.references:
        index = enriched_reference.image_index
        if enriched_reference.kind == "attribute":
            assert enriched_reference.attribute_id is not None
            assert enriched_reference.owner_entity_id is not None
            record = attributes.pop(enriched_reference.attribute_id, None)
            if record is None or (
                record.owner_entity_id != enriched_reference.owner_entity_id
                or record.owner_entity_id not in visual_entity_ids
                or record.image_path != enriched_reference.image_path
                or record.source_frame_index != enriched_reference.source_frame_index
            ):
                raise ValueError(
                    f"attribute provenance mismatch: {enriched_reference.attribute_id}"
                )
            if record.source_frame_index is None or record.image_path is None:
                raise ValueError("accepted attribute is missing frame/image provenance")
            artifact = _resolve_artifact(
                run_root / "subject_attributes", record.image_path
            )
            normalized.append(
                NormalizedVisualReference(
                    image_id=f"image_{index}",
                    image_index=index,
                    kind="attribute",
                    attribute_id=record.attribute_id,
                    owner_entity_id=record.owner_entity_id,
                    attribute_type=record.attribute_type,
                    image_path=record.image_path,
                    artifact_path=str(artifact),
                    source_frame_index=record.source_frame_index,
                    synthetic=record.final_selection == "completed",
                )
            )
            continue

        if enriched_reference.origin != "visual_run":
            raise ValueError("non-attribute enriched reference must originate in Visual")
        if enriched_reference.kind == "background":
            source = background_reference
            background_reference = None
        else:
            assert enriched_reference.entity_id is not None
            source = entity_references.pop(enriched_reference.entity_id, None)
        if source is None:
            raise ValueError(
                f"enriched Visual reference has no export match: "
                f"{enriched_reference.image_id}"
            )
        if (
            _dataset_reference_kind(source) != enriched_reference.kind
            or source.source_frame_index != enriched_reference.source_frame_index
        ):
            raise ValueError(
                f"enriched Visual provenance mismatch: {enriched_reference.image_id}"
            )
        artifact = _resolve_artifact(run_root, enriched_reference.image_path)
        normalized.append(
            NormalizedVisualReference(
                image_id=f"image_{index}",
                image_index=index,
                kind=enriched_reference.kind,
                entity_id=source.entity_id,
                image_path=enriched_reference.image_path,
                artifact_path=str(artifact),
                source_frame_index=source.source_frame_index,
                scope=source.scope,
                visible_region=source.visible_region,
                synthetic=source.synthetic,
            )
        )
    if entity_references or background_reference is not None or attributes:
        raise ValueError(
            f"enriched references do not cover Visual sample: {sample.sample_id}"
        )
    return NormalizedVisualSample(
        sample_id=sample.sample_id,
        clip_uid=enriched.clip_uid,
        target_video=sample.target_video,
        t2v_caption=sample.t2v_caption,
        r2v_instruction=enriched.enriched_instruction,
        references=normalized,
    )


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

    rows = iter(_read_jsonl(samples_path))
    try:
        first = next(rows)
    except StopIteration as exc:
        raise ValueError("canonical Visual samples.jsonl is empty") from exc
    schema_version = first.get("schema_version")
    if schema_version == PRODUCTION_SAMPLE_SCHEMA_VERSION:
        input_schema: VisualInputSchema = PRODUCTION_SAMPLE_SCHEMA_VERSION
        input_mode: VisualInputMode = "compacted_production"
        enriched_by_id: dict[str, EnrichedSample] = {}
    elif schema_version == SAMPLE_SCHEMA_VERSION:
        input_schema = SAMPLE_SCHEMA_VERSION
        input_mode = "single_run_export"
        enriched_by_id = _load_enriched_samples(runs_root)
    else:
        raise ValueError(f"unsupported Visual samples schema: {schema_version!r}")

    clips: list[VisualProductionClip] = []
    sample_ids: set[str] = set()
    clip_uids: set[str] = set()
    skipped: Counter[str] = Counter()
    canonical_count = 0
    for raw in chain((first,), rows):
        if raw.get("schema_version") != input_schema:
            raise ValueError("Visual samples.jsonl mixes incompatible schemas")
        if input_mode == "compacted_production":
            source_sample = ProductionSample.model_validate(raw)
            clip_uid = source_sample.clip_uid
            sample_id = source_sample.sample_id
            shard_id = source_sample.source.shard_id
            clip_path = (
                runs_root / shard_id / "clips" / clip_uid / "clip.json"
            ).resolve(strict=True)
            normalized_sample = NormalizedVisualSample(
                sample_id=sample_id,
                clip_uid=clip_uid,
                target_video=source_sample.target_video,
                t2v_caption=source_sample.t2v_caption,
                r2v_instruction=source_sample.r2v_instruction,
                references=[
                    _normalize_production_reference(
                        reference,
                        production_root=production_root,
                    )
                    for reference in source_sample.references
                ],
            )
        else:
            dataset_sample = DatasetSample.model_validate(raw)
            clip_uid = dataset_sample.sample_id
            sample_id = dataset_sample.sample_id
            shard_id = runs_root.name
            clip_path = (
                runs_root / "clips" / clip_uid / "clip.json"
            ).resolve(strict=True)
            enriched = enriched_by_id.get(sample_id)
            if enriched is None:
                normalized_sample = NormalizedVisualSample(
                    sample_id=sample_id,
                    clip_uid=clip_uid,
                    target_video=dataset_sample.target_video,
                    t2v_caption=dataset_sample.t2v_caption,
                    r2v_instruction=dataset_sample.r2v_instruction,
                    references=[
                        _normalize_dataset_reference(
                            reference,
                            image_index=index,
                            export_root=production_root,
                        )
                        for index, reference in enumerate(
                            dataset_sample.references, start=1
                        )
                    ],
                )
            else:
                normalized_sample = _normalize_enriched_sample(
                    dataset_sample,
                    enriched,
                    run_root=runs_root,
                )
        canonical_count += 1
        if sample_id in sample_ids:
            raise ValueError(f"duplicate canonical sample_id: {sample_id}")
        if clip_uid in clip_uids:
            raise ValueError(f"duplicate canonical clip_uid: {clip_uid}")
        sample_ids.add(sample_id)
        clip_uids.add(clip_uid)

        clip_path.relative_to(runs_root)
        clip = ClipRecord.model_validate_json(clip_path.read_text(encoding="utf-8"))
        if clip.clip_uid != clip_uid:
            raise ValueError("canonical sample does not match its clip.json")
        if normalized_sample.target_video != clip.source.video_path:
            raise ValueError(
                "canonical target_video differs from clip source video_path"
            )
        metadata = clip.source.metadata
        identity = derive_readable_clip_identity(
            clip_uid=clip_uid,
            shard_id=shard_id,
            source_relative_video_path=metadata.get("source_relative_video_path"),
            source_relative_source_video_path=metadata.get(
                "source_relative_source_video_path"
            ),
        )
        subject_references = [
            reference
            for reference in normalized_sample.references
            if reference.kind == "subject"
        ]
        for reference in subject_references:
            reference_path = Path(reference.artifact_path).resolve(strict=True)
            if not reference_path.is_file():
                raise ValueError("canonical subject reference is not a file")
        if not subject_references:
            skipped.update(["no_subject_reference"])
            continue
        clips.append(
            VisualProductionClip(
                identity=identity,
                sample=normalized_sample,
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
        visual_input_schema=input_schema,
        visual_input_mode=input_mode,
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
