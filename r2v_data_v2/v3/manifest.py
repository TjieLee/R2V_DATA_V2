from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from r2v_data_v2.manifest import iter_source_records, parse_source_record
from r2v_data_v2.naming import parse_clip_identity
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.schemas import ClipSource
from r2v_data_v2.v3.storage import RunStorage


@dataclass(frozen=True)
class ManifestStats:
    processed: int = 0
    skipped_existing: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SourceEvidence:
    caption_raw: str
    metadata: dict[str, object]


def source_evidence_by_video(
    dataset_json: str | Path,
) -> dict[str, SourceEvidence]:
    evidence: dict[str, SourceEvidence] = {}
    for raw in iter_source_records(dataset_json):
        try:
            parsed = parse_source_record(raw)
        except (TypeError, ValueError):
            continue
        evidence.setdefault(
            str(parsed.video_path),
            SourceEvidence(
                caption_raw=parsed.caption_raw,
                metadata=parsed.metadata,
            ),
        )
    return evidence


def build_manifest(
    config: V3Config,
    storage: RunStorage,
) -> ManifestStats:
    processed = skipped_existing = failed = 0
    for source_index, raw in enumerate(iter_source_records(config.dataset_json)):
        clip_uid = None
        video_path = None
        try:
            parsed = parse_source_record(raw)
            video_path = parsed.video_path
            if not video_path.is_file():
                raise FileNotFoundError(
                    f"source video does not exist: {video_path}"
                )
            identity = parse_clip_identity(video_path)
            clip_uid = identity.clip_uid
            existed = storage.clip_path(clip_uid).is_file()
            storage.create_clip(
                clip_uid=clip_uid,
                source=ClipSource(
                    video_path=str(video_path),
                    parent_video_id=identity.parent_video_id,
                    clip_suffix=identity.clip_suffix,
                ),
            )
            if existed:
                skipped_existing += 1
            else:
                processed += 1
        except Exception as exc:  # noqa: BLE001 - isolate malformed source records
            details: dict[str, object] = {"source_index": source_index}
            if video_path is not None:
                details["video_path"] = str(video_path)
            storage.append_failure(
                clip_uid=clip_uid,
                stage="manifest",
                reason=str(exc),
                details=details,
            )
            failed += 1
    stats = ManifestStats(
        processed=processed,
        skipped_existing=skipped_existing,
        failed=failed,
    )
    storage.update_stage_counts("manifest", stats.to_dict())
    return stats
