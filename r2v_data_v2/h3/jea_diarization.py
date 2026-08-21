from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from r2v_data_v2.h3.diarization_binding import (
    BoundDiarizationSegment,
    DiarizationInventory,
    RawDiarizationSegment,
)
from r2v_data_v2.h3.schemas import SchemaModel
from r2v_data_v2.h3.visual_production_source import VisualProductionInventory


class JEAReadableDiarizationTarget(SchemaModel):
    schema_version: Literal["r2v.h3.jea_diarization_target.1"] = (
        "r2v.h3.jea_diarization_target.1"
    )
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    source_audio_path: str
    source_sample_rate_hz: int = Field(gt=0)
    target_video_path: str
    target_audio_binding_path: str


class JEAReadableDiarizationSegment(SchemaModel):
    schema_version: Literal["r2v.h3.jea_diarization_segment.1"] = (
        "r2v.h3.jea_diarization_segment.1"
    )
    clip_uid: str
    clip_display_path: str
    media_collection_relpath: str
    media_collection_name: str
    episode_name: str
    clip_name: str
    shard_id: str
    segment_id: str
    speaker_cluster_id: str
    entity_id: str | None = None
    entity_occurrence_id: str | None = None
    source_audio_path: str
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(gt=0)
    source_sample_rate_hz: int = Field(gt=0)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    raw_schema_version: Literal["r2v.h3.diarization_segment.2"] = (
        "r2v.h3.diarization_segment.2"
    )
    bound_schema_version: Literal["r2v.h3.diarization_bound_segment.1"] = (
        "r2v.h3.diarization_bound_segment.1"
    )
    mapping_policy_version: Literal["h3_diarizen_sparse_anchor_policy_v1"] = (
        "h3_diarizen_sparse_anchor_policy_v1"
    )
    segmentation_changed: Literal[False] = False
    numeric_mapping_thresholds_changed: Literal[False] = False


class JEAReadableDiarizationSummary(SchemaModel):
    schema_version: Literal["r2v.h3.jea_diarization_summary.1"] = (
        "r2v.h3.jea_diarization_summary.1"
    )
    target_count: int = Field(ge=0)
    segment_count: int = Field(ge=0)
    media_collection_count: int = Field(ge=0)
    segmentation_changed: Literal[False] = False
    numeric_mapping_thresholds_changed: Literal[False] = False


def _read(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write(path: Path, values: Sequence[SchemaModel]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                value.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def publish_readable_diarization_metadata(
    *,
    visual_inventory: VisualProductionInventory,
    diarization_root: Path,
) -> JEAReadableDiarizationSummary:
    root = diarization_root.expanduser().resolve(strict=True)
    inventory = DiarizationInventory.model_validate_json(
        (root / "inventory.json").read_text(encoding="utf-8")
    )
    raw = [
        RawDiarizationSegment.model_validate(row)
        for row in _read(root / "raw_segments.jsonl")
    ]
    bound = [
        BoundDiarizationSegment.model_validate(row)
        for row in _read(root / "bound_segments.jsonl")
    ]
    bound_by_key = {(item.target_clip_uid, item.segment_id): item for item in bound}
    if len(bound_by_key) != len(bound) or set(bound_by_key) != {
        (item.target_clip_uid, item.segment_id) for item in raw
    }:
        raise ValueError("DiariZen raw and bound segment inventories differ")
    identity_by_clip = {
        item.identity.clip_uid: item.identity for item in visual_inventory.clips
    }
    targets: list[JEAReadableDiarizationTarget] = []
    for target in inventory.targets:
        identity = identity_by_clip[target.target_clip_uid]
        targets.append(
            JEAReadableDiarizationTarget(
                **identity.model_dump(mode="python"),
                source_audio_path=target.source_audio_path,
                source_sample_rate_hz=target.source_sample_rate_hz,
                target_video_path=target.target_video_path,
                target_audio_binding_path=target.target_audio_binding_path,
            )
        )
    segments: list[JEAReadableDiarizationSegment] = []
    for source in raw:
        identity = identity_by_clip[source.target_clip_uid]
        mapped = bound_by_key[(source.target_clip_uid, source.segment_id)]
        if (
            mapped.source_start_sample != source.source_start_sample
            or mapped.source_end_sample != source.source_end_sample
            or mapped.speaker_cluster_id != source.speaker_cluster_id
        ):
            raise ValueError("bound DiariZen segment changed its raw sample extent")
        segments.append(
            JEAReadableDiarizationSegment(
                **identity.model_dump(mode="python"),
                segment_id=source.segment_id,
                speaker_cluster_id=source.speaker_cluster_id,
                entity_id=mapped.entity_id,
                entity_occurrence_id=mapped.entity_occurrence_id,
                source_audio_path=source.source_audio_path,
                source_start_sample=source.source_start_sample,
                source_end_sample=source.source_end_sample,
                source_sample_rate_hz=source.source_sample_rate_hz,
                start_time=source.start_time,
                end_time=source.end_time,
            )
        )
    targets.sort(key=lambda item: item.clip_display_path)
    segments.sort(
        key=lambda item: (
            item.clip_display_path,
            item.source_start_sample,
            item.segment_id,
        )
    )
    _write(root / "readable_targets.jsonl", targets)
    _write(root / "readable_segments.jsonl", segments)
    summary = JEAReadableDiarizationSummary(
        target_count=len(targets),
        segment_count=len(segments),
        media_collection_count=len({item.media_collection_relpath for item in targets}),
    )
    (root / "readable_summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
