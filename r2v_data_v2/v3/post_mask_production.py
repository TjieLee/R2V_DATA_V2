"""Model-free, read-only Stage2 ingestion into ordinary writable V3 storage.

Callers must hold the shard mutation lock across initialization and hydration.
Only this module's staging directories may be rebuilt; a published clip is never
replaced. Physical placement belongs to the caller, not the semantic V3 config.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import r2v_data_v2.v3.config as config_module
from r2v_data_v2.reconciliation import write_json_atomic
from r2v_data_v2.v3.config import RuntimeConfig, V3Config
from r2v_data_v2.v3.schemas import (
    ClipRecord,
    SampledFramesArtifact,
    TrackedMasksArtifact,
)
from r2v_data_v2.v3.storage import RunStorage

_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHARD = re.compile(r"shard-[0-9]{9}-[0-9]{9}\.jsonl")
_PROVENANCE = ".post_mask_hydration.json"


def _component(value: object) -> str:
    if not isinstance(value, str) or not _COMPONENT.fullmatch(value):
        raise ValueError("unsafe clip/shard identity")
    return value


def _beneath(root: Path, relative: str) -> Path:
    path = Path(relative)
    if root.is_symlink():
        raise ValueError("artifact owner must not be a symlink")
    if path.is_absolute() or not relative or ".." in path.parts:
        raise ValueError("unsafe artifact path")
    result = root / path
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes its owner")
    for parent in (result, *result.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("symlink artifact is not a frozen regular file")
    return result


def enumerate_shards(entity_mask_root: Path) -> tuple[Path, ...]:
    """Return all canonical parts/shard-*.jsonl files in lexical order."""
    root = Path(entity_mask_root).resolve()
    paths = []
    for path in sorted((root / "parts").glob("shard-*.jsonl")):
        if not _SHARD.fullmatch(path.name):
            raise ValueError("invalid canonical shard filename")
        checked = _beneath(root, path.relative_to(root).as_posix())
        if not checked.is_file():
            raise ValueError("canonical shard is not a file")
        paths.append(checked)
    return tuple(paths)


@dataclass(frozen=True)
class ShardPaths:
    shard_path: Path
    run_root: Path
    export_root: Path
    state_root: Path

    @classmethod
    def for_shard(
        cls,
        post_mask_root: Path,
        shard_path: Path,
        *,
        runs_root: Path | None = None,
        exports_root: Path | None = None,
    ) -> ShardPaths:
        """Stable paths: optional roots are campaign-wide, never worker-local."""
        state = Path(post_mask_root).resolve()
        shard = Path(shard_path).absolute()
        if not _SHARD.fullmatch(shard.name):
            raise ValueError("invalid canonical shard filename")
        tag = _component(state.name)
        writable = config_module.ALLOWED_WRITABLE_ROOT
        runs = runs_root or writable / "r2v_v3_runs/production/jea_motion_v1" / tag
        exports = (
            exports_root
            or writable / "r2v_v3_exports/production/jea_motion_v1" / tag / "shards"
        )
        paths = cls(
            shard,
            Path(runs).resolve() / shard.stem,
            Path(exports).resolve() / shard.stem,
            state / "shards" / shard.stem,
        )
        for path in (paths.run_root, paths.export_root, paths.state_root):
            if not path.is_relative_to(writable.resolve()):
                raise ValueError(
                    "Post-Mask writes must stay inside allowed writable root"
                )
        return paths

    @property
    def exclusions_path(self) -> Path:
        return self.state_root / "exclusions.jsonl"

    @property
    def input_failures_path(self) -> Path:
        return self.state_root / "input_failures.jsonl"

    @property
    def staging_root(self) -> Path:
        # Same filesystem as destination for atomic directory publication.
        return self.run_root / ".post_mask_staging"

    @property
    def identity_path(self) -> Path:
        return self.state_root / "identity.json"


@dataclass(frozen=True)
class HydrationResult:
    clip_uids: tuple[str, ...]
    ready: int
    excluded: int
    corrupt: int


def prepare_shard_config(base_config: V3Config, paths: ShardPaths) -> V3Config:
    """Preserve Visual policy/models; replace roots and execution-only settings.

    Local execution adapters receive actual devices, concurrency and timeouts
    separately from the accepted base config/launcher. RuntimeConfig defaults
    and neutral cuda/0 are stable placeholders, not execution requests. Normal
    V3 config/fingerprint validation remains unchanged.
    """
    config = replace(
        base_config,
        run_root=paths.run_root,
        export_root=paths.export_root,
        sam3=replace(base_config.sam3, device="cuda"),
        remove=replace(base_config.remove, device="cuda"),
        reference_edit=replace(base_config.reference_edit, cuda_visible_devices="0"),
        runtime=RuntimeConfig(),
    )
    config.validate()
    return config


def _digest(path: Path) -> str:
    # Small JSON/JSONL identity artifacts only; never JPEG/payload hashing.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_identity(config: V3Config, paths: ShardPaths) -> dict[str, Any]:
    """Cheap campaign identity; canonical bytes are bound only on shard entry."""
    return {
        "canonical_shard": str(paths.shard_path),
        "config_hash": config.fingerprint(),
        "subject_attributes": json.loads(json.dumps(asdict(config.subject_attributes))),
    }


def _identity(config: V3Config, paths: ShardPaths) -> dict[str, Any]:
    return {
        **_semantic_identity(config, paths),
        "shard_sha256": _digest(paths.shard_path),
    }


def initialize_shard(
    base_config: V3Config, paths: ShardPaths, *, git_commit: str
) -> RunStorage:
    """Initialize strict V3 identity, retaining a historical commit on restart."""
    frozen_root = paths.shard_path.parent.parent.resolve()
    for destination in (paths.run_root, paths.export_root, paths.state_root):
        resolved = destination.resolve()
        if resolved.is_relative_to(frozen_root) or frozen_root.is_relative_to(resolved):
            raise ValueError("Post-Mask destination overlaps frozen input")
        if not resolved.is_relative_to(config_module.ALLOWED_WRITABLE_ROOT.resolve()):
            raise ValueError("Post-Mask writes must stay inside allowed writable root")
    config = prepare_shard_config(base_config, paths)
    expected = _identity(config, paths)
    if (
        paths.identity_path.exists()
        and json.loads(paths.identity_path.read_text()) != expected
    ):
        raise ValueError("Post-Mask shard/config identity mismatch")
    storage = RunStorage(config)
    if storage.run_path.exists():
        git_commit = storage.read_run().git_commit
        # An interrupted fresh initialization may leave only run.json.
        if not paths.identity_path.exists() and any(
            path.name != "run.json" for path in storage.root.iterdir()
        ):
            raise ValueError("populated Post-Mask run is missing identity sidecar")
    storage.initialize(git_commit=git_commit)
    write_json_atomic(paths.identity_path, expected)
    return storage


def _write_inventory(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _reference_paths(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("_path") and isinstance(item, str):
                yield item
            else:
                yield from _reference_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _reference_paths(item)


def _validate_input(
    root: Path, shard: Path, row: dict[str, Any]
) -> tuple[Path, dict[str, Any], tuple[Path, ...]]:
    uid = _component(row.get("clip_uid"))
    index = row.get("source_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("invalid source_index")
    artifact = row.get("artifact_root")
    if not isinstance(artifact, str):
        raise TypeError("ready row requires artifact_root")
    if artifact != f"artifacts/{shard.stem}/{uid}":
        raise ValueError("artifact_root does not match canonical shard/clip layout")
    workspace = _beneath(root, artifact)
    clip_dir = _beneath(workspace, f"run/clips/{uid}")
    if not clip_dir.is_dir():
        raise ValueError("missing Stage2 clip directory")
    manifest_paths = [
        _beneath(clip_dir, relative)
        for relative in ("clip.json", "frames/frames.json", "masks.rle.json")
    ]
    if any(not path.is_file() for path in manifest_paths):
        raise ValueError("missing or non-regular Stage2 manifest")
    required = set(manifest_paths)
    clip = ClipRecord.model_validate_json(manifest_paths[0].read_text())
    frames = SampledFramesArtifact.model_validate_json(manifest_paths[1].read_text())
    masks = TrackedMasksArtifact.model_validate_json(manifest_paths[2].read_text())
    if (
        clip.clip_uid != uid
        or frames.clip_uid != uid
        or masks.clip_uid != uid
        or clip.source.source_index != index
    ):
        raise ValueError("Stage2 target identity mismatch")
    if (
        clip.annotation is None
        or clip.annotation.status != "ready"
        or not clip.annotation.entities
        or clip.coverage is None
        or not clip.coverage.passed
        or row.get("coverage_passed") is not True
    ):
        raise ValueError("ready row lacks ready annotation/coverage")
    background = clip.references.background
    if background is None or row["status"] != {
        "none": "ready_no_background",
        "rejected": "ready_background_rejected",
        "pending_remove": "ready_background_pending_remove",
    }.get(background.status):
        raise ValueError("Stage2 background/status mismatch")
    if row.get("background_status", background.status) != background.status:
        raise ValueError("Stage2 background identity mismatch")
    video = Path(clip.source.video_path)
    if (
        not video.is_absolute()
        or not video.resolve().is_relative_to(
            config_module.ALLOWED_DATASET_ROOT.resolve()
        )
        or not video.is_file()
    ):
        raise ValueError("missing or unsafe target video")
    if (masks.width, masks.height) != (frames.width, frames.height):
        raise ValueError("frames/masks dimension mismatch")
    if set(masks.entities) != {entity.entity_id for entity in clip.annotation.entities}:
        raise ValueError("annotation/masks entity identity mismatch")
    for frame in frames.frames:
        path = _beneath(clip_dir, frame.image_path)
        if not path.is_file():
            raise ValueError("missing sampled frame")
        required.add(path)
    for relative in _reference_paths(clip.references.model_dump(mode="json")):
        # Existing schemas use both clip-relative frame and run-relative paths.
        if relative.startswith("clips/"):
            prefix = f"clips/{uid}/"
            if not relative.startswith(prefix):
                raise ValueError("reference escapes source clip")
            relative = relative[len(prefix) :]
        path = _beneath(clip_dir, relative)
        if not path.is_file():
            raise ValueError("missing reference asset")
        required.add(path)
    provenance = {
        "row": row,
        "artifact_root": str(workspace),
        "manifests": {
            str(path.relative_to(clip_dir)): _digest(path) for path in manifest_paths
        },
    }
    return clip_dir, provenance, tuple(sorted(required))


def _make_staging_writable(root: Path) -> None:
    """Restore owner permissions only on our independent copied staging tree."""
    for directory, directories, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        if parent.is_symlink() or any(
            (parent / name).is_symlink() for name in (*directories, *files)
        ):
            raise ValueError("owned hydration staging contains a symlink")
        parent.chmod(parent.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        for name in files:
            path = parent / name
            path.chmod(path.stat().st_mode | stat.S_IRUSR | stat.S_IWUSR)


def hydrate_shard(
    storage: RunStorage, *, entity_mask_root: Path, shard_path: Path, paths: ShardPaths
) -> HydrationResult:
    """Isolate bad ready rows, copy good clips atomically, preserve published work.

    Returns input-order eligible clip IDs and deterministic row counts. Rewrites
    inventories atomically, so restart never duplicates input-failure records.
    Requires initialize_shard and an external exclusive shard lock.
    """
    root = Path(entity_mask_root).resolve()
    shard = Path(shard_path).absolute()
    if shard not in enumerate_shards(root) or shard != paths.shard_path:
        raise ValueError("not a canonical Stage2 shard")
    if storage.root != paths.run_root or _identity(storage.config, paths) != json.loads(
        paths.identity_path.read_text()
    ):
        raise ValueError("Post-Mask source/config identity mismatch")
    for destination in (paths.run_root, paths.export_root, paths.state_root):
        if destination.resolve().is_relative_to(root):
            raise ValueError("Post-Mask destination overlaps frozen input")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(shard.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("Stage2 row must be an object")
            rows.append(row)
        except (ValueError, TypeError):
            rows.append({"status": "invalid_row", "line_number": line_number})
    uids = Counter(
        row.get("clip_uid") for row in rows if isinstance(row.get("clip_uid"), str)
    )
    indices = Counter(
        row.get("source_index")
        for row in rows
        if isinstance(row.get("source_index"), int)
        and not isinstance(row.get("source_index"), bool)
    )
    excluded, failures, ready = [], [], []
    for row in rows:
        record = {
            key: row.get(key)
            for key in ("source_index", "clip_uid", "status", "reason", "artifact_root")
        }
        record["canonical_shard"] = str(shard)
        if "line_number" in row:
            record["line_number"] = row["line_number"]
        status = row.get("status")
        if isinstance(status, str) and (
            status == "coverage_rejected" or status.startswith(("skipped_", "failed_"))
        ):
            excluded.append(record)
            continue
        try:
            if not isinstance(status, str) or not status.startswith("ready_"):
                raise ValueError("invalid Stage2 row status")
            uid = _component(row.get("clip_uid"))
            index = row.get("source_index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or uids[uid] > 1
                or indices[index] > 1
            ):
                raise ValueError("invalid or duplicate Stage2 row identity")
            source, provenance, required = _validate_input(root, shard, row)
            provenance["canonical_shard"] = str(shard)
            destination = _beneath(storage.root, f"clips/{uid}")
            marker = destination / _PROVENANCE
            if destination.exists():
                if not marker.is_file() or json.loads(marker.read_text()) != provenance:
                    raise ValueError("published clip hydration provenance mismatch")
                current = storage.read_clip(uid)
                original = ClipRecord.model_validate_json(
                    (source / "clip.json").read_text()
                )
                if current.clip_uid != uid or current.source != original.source:
                    raise ValueError("published clip source identity mismatch")
            else:
                staging = _beneath(paths.staging_root, uid)
                if staging.exists():
                    _make_staging_writable(staging)
                    shutil.rmtree(staging)
                staging.mkdir(parents=True)
                for path in required:
                    target = staging / path.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                _make_staging_writable(staging)
                write_json_atomic(staging / _PROVENANCE, provenance)
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging.replace(destination)
            ready.append(uid)
        except (ValueError, TypeError, OSError) as exc:
            failures.append({**record, "failure_reason": str(exc), "stage": "hydrate"})
    _write_inventory(paths.exclusions_path, excluded)
    _write_inventory(paths.input_failures_path, failures)
    return HydrationResult(tuple(ready), len(ready), len(excluded), len(failures))
