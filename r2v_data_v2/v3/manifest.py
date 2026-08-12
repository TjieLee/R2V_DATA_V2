from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from decimal import Decimal
from itertools import islice
from pathlib import Path
from typing import Any

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.manifest import iter_source_records, parse_source_record
from r2v_data_v2.naming import parse_clip_identity
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.schemas import ClipRecord, ClipSource
from r2v_data_v2.v3.storage import RunStorage


@dataclass(frozen=True)
class ManifestStats:
    processed: int = 0
    skipped_existing: int = 0
    failed: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class SelectedSource:
    source_index: int
    video_path: Path
    clip_uid: str
    parent_video_id: str
    clip_suffix: str
    caption_raw: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class SelectionDiagnostics:
    source_records_scanned: int = 0
    parents_discovered: int = 0
    filesystem_checks: int = 0
    missing_selected_candidates: int = 0


_FIXED_SELECTION_FIELDS = (
    "source_index",
    "clip_uid",
    "parent_video_id",
    "video_path",
    "caption_raw",
    "metadata",
    "clip_suffix",
)


def _parse_selected_source(
    source_index: int,
    raw: dict[str, object],
    *,
    check_video_exists: bool = True,
) -> SelectedSource:
    parsed = parse_source_record(raw)
    if check_video_exists and not _source_video_is_file(parsed.video_path):
        raise FileNotFoundError(f"source video does not exist: {parsed.video_path}")
    identity = parse_clip_identity(parsed.video_path)
    return SelectedSource(
        source_index=source_index,
        video_path=parsed.video_path,
        clip_uid=identity.clip_uid,
        parent_video_id=identity.parent_video_id,
        clip_suffix=identity.clip_suffix,
        caption_raw=parsed.caption_raw,
        metadata=parsed.metadata,
    )


def _source_video_is_file(video_path: Path) -> bool:
    return video_path.is_file()


def _fixed_metadata_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("fixed selection metadata numbers must be finite")
        if value == value.to_integral_value():
            return int(value)
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("fixed selection metadata numbers must be finite")
        return converted
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("fixed selection metadata numbers must be finite")
        return value
    if isinstance(value, list):
        return [_fixed_metadata_value(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _fixed_metadata_value(item) for key, item in value.items()}
    raise TypeError("fixed selection metadata must contain only JSON values")


def _parse_fixed_selected_source(raw: dict[str, Any]) -> SelectedSource:
    missing = [name for name in _FIXED_SELECTION_FIELDS if name not in raw]
    if missing:
        raise ValueError(f"fixed selection record is missing fields: {missing}")
    source_index = raw["source_index"]
    if (
        not isinstance(source_index, int)
        or isinstance(source_index, bool)
        or source_index < 0
    ):
        raise ValueError("fixed selection source_index must be a non-negative integer")
    strings: dict[str, str] = {}
    for name in ("clip_uid", "parent_video_id", "video_path", "caption_raw", "clip_suffix"):
        value = raw[name]
        if not isinstance(value, str) or (name != "caption_raw" and not value):
            raise ValueError(f"fixed selection {name} must be a valid string")
        strings[name] = value
    source_path = Path(strings["video_path"]).expanduser()
    if not source_path.is_absolute():
        raise ValueError("fixed selection video_path must be absolute")
    video_path = source_path.resolve(strict=False)
    dataset_root = config_module.ALLOWED_DATASET_ROOT.resolve(strict=False)
    if dataset_root not in video_path.parents:
        raise ValueError("fixed selection video_path must be inside public dataset")
    identity = parse_clip_identity(video_path)
    if strings["clip_uid"] != identity.clip_uid:
        raise ValueError("fixed selection clip_uid does not match video_path")
    if strings["parent_video_id"] != identity.parent_video_id:
        raise ValueError("fixed selection parent_video_id does not match video_path")
    if strings["clip_suffix"] != identity.clip_suffix:
        raise ValueError("fixed selection clip_suffix does not match video_path")
    raw_metadata = raw["metadata"]
    if not isinstance(raw_metadata, dict) or not all(
        isinstance(key, str) for key in raw_metadata
    ):
        raise ValueError("fixed selection metadata must be an object with string keys")
    metadata = _fixed_metadata_value(raw_metadata)
    assert isinstance(metadata, dict)
    if not _source_video_is_file(video_path):
        raise FileNotFoundError(f"source video does not exist: {video_path}")
    return SelectedSource(
        source_index=source_index,
        video_path=video_path,
        clip_uid=identity.clip_uid,
        parent_video_id=identity.parent_video_id,
        clip_suffix=identity.clip_suffix,
        caption_raw=strings["caption_raw"],
        metadata=dict(metadata),
    )


def _fixed_selected_sources(
    config: V3Config,
) -> tuple[list[SelectedSource], SelectionDiagnostics]:
    selection_manifest = config.source.selection_manifest
    if selection_manifest is None:
        raise ValueError("source.selection_manifest is required for fixed selection")
    records = iter_source_records(selection_manifest)
    if config.source.limit is not None:
        records = islice(records, config.source.limit)
    selected: list[SelectedSource] = []
    seen_clip_uids: set[str] = set()
    for raw in records:
        item = _parse_fixed_selected_source(raw)
        if item.clip_uid in seen_clip_uids:
            raise ValueError(
                f"duplicate clip_uid in fixed selection: {item.clip_uid}"
            )
        seen_clip_uids.add(item.clip_uid)
        selected.append(item)
    return selected, SelectionDiagnostics(
        source_records_scanned=len(selected),
        parents_discovered=len({item.parent_video_id for item in selected}),
        filesystem_checks=len(selected),
    )


def _random_selected_sources(
    config: V3Config,
    storage: RunStorage,
) -> tuple[list[SelectedSource], int, SelectionDiagnostics]:
    by_parent: dict[str, list[SelectedSource]] = {}
    failed = 0
    source_records_scanned = 0
    seen_clip_uids: set[str] = set()
    for source_index, raw in enumerate(iter_source_records(config.dataset_json)):
        if source_index < config.source.start_index:
            continue
        source_records_scanned += 1
        try:
            selected = _parse_selected_source(
                source_index,
                raw,
                check_video_exists=False,
            )
            if selected.clip_uid in seen_clip_uids:
                raise ValueError(
                    f"duplicate clip identity in source dataset: {selected.clip_uid}"
                )
            seen_clip_uids.add(selected.clip_uid)
            by_parent.setdefault(selected.parent_video_id, []).append(selected)
        except Exception as exc:  # noqa: BLE001 - isolate malformed source records
            storage.append_failure(
                clip_uid=None,
                stage="manifest",
                reason=str(exc),
                details={"source_index": source_index},
            )
            failed += 1

    assert config.source.random_seed is not None
    assert config.source.limit is not None
    rng = random.Random(config.source.random_seed)
    parent_ids = sorted(by_parent)
    rng.shuffle(parent_ids)
    for parent_id in parent_ids:
        rng.shuffle(by_parent[parent_id])

    eligible_count = sum(
        min(len(by_parent[parent_id]), config.source.max_clips_per_parent)
        for parent_id in parent_ids
    )
    if eligible_count < config.source.limit:
        raise ValueError(
            "source.limit cannot be satisfied under source.max_clips_per_parent: "
            f"requested {config.source.limit}, eligible {eligible_count}"
        )

    chosen: list[SelectedSource] = []
    filesystem_checks = 0
    missing_selected_candidates = 0
    for parent_id in parent_ids:
        accepted_for_parent = 0
        for candidate in by_parent[parent_id]:
            if accepted_for_parent >= config.source.max_clips_per_parent:
                break
            filesystem_checks += 1
            if not _source_video_is_file(candidate.video_path):
                missing_selected_candidates += 1
                failed += 1
                storage.append_failure(
                    clip_uid=candidate.clip_uid,
                    stage="manifest",
                    reason=f"source video does not exist: {candidate.video_path}",
                    details={
                        "source_index": candidate.source_index,
                        "video_path": str(candidate.video_path),
                    },
                )
                continue
            chosen.append(candidate)
            accepted_for_parent += 1
            if len(chosen) >= config.source.limit:
                break
        if len(chosen) >= config.source.limit:
            break
    if len(chosen) < config.source.limit:
        raise ValueError(
            "source.limit cannot be satisfied after validating selected source "
            f"videos: requested {config.source.limit}, available {len(chosen)}"
        )
    diagnostics = SelectionDiagnostics(
        source_records_scanned=source_records_scanned,
        parents_discovered=len(parent_ids),
        filesystem_checks=filesystem_checks,
        missing_selected_candidates=missing_selected_candidates,
    )
    return chosen, failed, diagnostics


def _write_selection_provenance(
    config: V3Config,
    storage: RunStorage,
    selected: list[SelectedSource],
    diagnostics: SelectionDiagnostics,
) -> None:
    provenance: dict[str, object] = {
        "selection_mode": config.source.selection_mode,
        "random_seed": config.source.random_seed,
        "max_clips_per_parent": config.source.max_clips_per_parent,
        "requested_limit": config.source.limit,
        "selected_count": len(selected),
        **asdict(diagnostics),
        "selected": [
            {
                "source_index": item.source_index,
                "clip_uid": item.clip_uid,
                "parent_video_id": item.parent_video_id,
            }
            for item in selected
        ],
    }
    if config.source.selection_manifest is not None:
        provenance["selection_manifest"] = str(
            config.source.selection_manifest.expanduser().resolve(strict=False)
        )
    write_json_atomic(
        storage.root / "source_selection.json",
        provenance,
    )


def _create_selected_clip(storage: RunStorage, selected: SelectedSource) -> bool:
    existed = storage.clip_path(selected.clip_uid).is_file()
    storage.create_clip(
        clip_uid=selected.clip_uid,
        source=ClipSource(
            video_path=str(selected.video_path),
            parent_video_id=selected.parent_video_id,
            clip_suffix=selected.clip_suffix,
            source_index=selected.source_index,
            caption_raw=selected.caption_raw,
            metadata=selected.metadata,
        ),
    )
    return existed


def _ordered_existing_clips(run_root: Path) -> list[ClipRecord]:
    clips = [
        ClipRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((run_root / "clips").glob("*/clip.json"))
    ]
    by_uid = {clip.clip_uid: clip for clip in clips}
    if len(by_uid) != len(clips):
        raise ValueError("existing V3 run contains duplicate clip_uid records")
    provenance_path = run_root / "source_selection.json"
    if not provenance_path.is_file():
        return sorted(clips, key=lambda clip: (clip.source.source_index, clip.clip_uid))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    selected = provenance.get("selected") if isinstance(provenance, dict) else None
    if not isinstance(selected, list):
        raise TypeError("source_selection.json selected must be a list")
    ordered_uids: list[str] = []
    seen_uids: set[str] = set()
    for item in selected:
        clip_uid = item.get("clip_uid") if isinstance(item, dict) else None
        if not isinstance(clip_uid, str) or not clip_uid:
            raise ValueError("source_selection.json contains an invalid clip_uid")
        if clip_uid in seen_uids:
            raise ValueError("source_selection.json contains duplicate clip_uid")
        seen_uids.add(clip_uid)
        ordered_uids.append(clip_uid)
    if set(ordered_uids) != set(by_uid):
        raise ValueError("source_selection.json does not match existing clip records")
    return [by_uid[clip_uid] for clip_uid in ordered_uids]


def export_fixed_selection_manifest(
    run_root: str | Path,
    destination: str | Path,
) -> Path:
    """Export one reusable fixed-selection JSON without modifying the source run."""

    source_root = Path(run_root).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("run_root must be a directory")
    output = Path(destination).expanduser().resolve(strict=False)
    if output.suffix.lower() != ".json":
        raise ValueError("fixed selection export must use a .json path")
    if output == source_root or source_root in output.parents:
        raise ValueError("fixed selection export must remain outside source run_root")
    records = [
        {
            "source_index": clip.source.source_index,
            "clip_uid": clip.clip_uid,
            "parent_video_id": clip.source.parent_video_id,
            "video_path": clip.source.video_path,
            "caption_raw": clip.source.caption_raw,
            "metadata": clip.source.metadata,
            "clip_suffix": clip.source.clip_suffix,
        }
        for clip in _ordered_existing_clips(source_root)
    ]
    write_json_atomic(
        output,
        {
            "schema_version": "r2v.v3.fixed-selection.1",
            "records": records,
        },
    )
    return output


def _materialize_selected_sources(
    storage: RunStorage,
    selected_sources: list[SelectedSource],
    *,
    failed: int,
) -> ManifestStats:
    processed = skipped_existing = 0
    for selected in selected_sources:
        try:
            if _create_selected_clip(storage, selected):
                skipped_existing += 1
            else:
                processed += 1
        except Exception as exc:  # noqa: BLE001 - isolate source conflicts
            storage.append_failure(
                clip_uid=selected.clip_uid,
                stage="manifest",
                reason=str(exc),
                details={
                    "source_index": selected.source_index,
                    "video_path": str(selected.video_path),
                },
            )
            failed += 1
    stats = ManifestStats(
        processed=processed,
        skipped_existing=skipped_existing,
        failed=failed,
    )
    storage.update_stage_counts("manifest", stats.to_dict())
    return stats


def build_manifest(
    config: V3Config,
    storage: RunStorage,
) -> ManifestStats:
    if config.source.selection_mode in {
        "parent_stratified_random_v1",
        "fixed_selection_v1",
    }:
        if config.source.selection_mode == "parent_stratified_random_v1":
            selected_sources, failed, diagnostics = _random_selected_sources(
                config,
                storage,
            )
        else:
            selected_sources, diagnostics = _fixed_selected_sources(config)
            failed = 0
        _write_selection_provenance(
            config,
            storage,
            selected_sources,
            diagnostics,
        )
        return _materialize_selected_sources(
            storage,
            selected_sources,
            failed=failed,
        )

    processed = skipped_existing = failed = 0
    selected = 0
    for source_index, raw in enumerate(iter_source_records(config.dataset_json)):
        if source_index < config.source.start_index:
            continue
        if config.source.limit is not None and selected >= config.source.limit:
            break
        selected += 1
        clip_uid = None
        video_path = None
        try:
            selected_source = _parse_selected_source(source_index, raw)
            video_path = selected_source.video_path
            clip_uid = selected_source.clip_uid
            existed = _create_selected_clip(storage, selected_source)
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
