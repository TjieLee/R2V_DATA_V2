from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path

from r2v_data_v2.manifest import iter_source_records, parse_source_record
from r2v_data_v2.naming import parse_clip_identity
from r2v_data_v2.reconciliation import write_json_atomic
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
    write_json_atomic(
        storage.root / "source_selection.json",
        {
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
        },
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


def build_manifest(
    config: V3Config,
    storage: RunStorage,
) -> ManifestStats:
    if config.source.selection_mode == "parent_stratified_random_v1":
        selected_sources, failed, diagnostics = _random_selected_sources(
            config,
            storage,
        )
        _write_selection_provenance(
            config,
            storage,
            selected_sources,
            diagnostics,
        )
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
