"""Clip-scoped cleanup of disposable Subject Attribute epoch binaries.

Only an existing ready clip outcome and its frozen clip plan authorize cleanup.
The phase plans supply job-to-clip provenance; binary trees are never searched
for owners, and neither images nor masks are opened.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from r2v_data_v2.v3.post_mask_epoch_state import file_lock

_GROUP_NAME = re.compile(r"group-[0-9]{6}\Z")
_SHARD_NAME = re.compile(r"shard-[0-9]{9}-[0-9]{9}\Z")
_JOB_ID = re.compile(r"[0-9a-f]{64}\Z")
_STATS = (
    "groups_seen",
    "terminal_clips_seen",
    "sam_job_dirs_cleaned",
    "sam_files_cleaned",
    "completion_job_dirs_cleaned",
    "completion_pngs_cleaned",
    "already_missing",
    "failures",
)


@dataclass
class BinaryJobIds:
    """The already-planned disposable jobs of one canonical clip."""

    sam: set[str] = field(default_factory=set)
    completion: set[str] = field(default_factory=set)


def _stats() -> dict[str, int]:
    return {name: 0 for name in _STATS}


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"symlinked durable JSON is unsafe: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid durable JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"durable JSON is not an object: {path}")
    return payload


def _safe_clip_uid(value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError("invalid phase plan clip_uid")
    if Path(value).name != value or "\\" in value:
        raise ValueError("unsafe phase plan clip_uid")
    return value


def load_binary_job_index(group_root: Path) -> dict[tuple[str, str], BinaryJobIds]:
    """Read each existing phase plan once; do not inspect binary contents.

    Existing plan digest/identity values are neither changed nor recomputed.
    The job ID syntax check is solely to keep deletion below one job directory.
    """
    # Lazy import avoids an import cycle: the SA runner uses this helper.
    from r2v_data_v2.v3.post_mask_epoch_subject_attributes import (
        SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_JOB,
        SUBJECT_ATTRIBUTE_COMPLETION_SAM_JOB,
        SUBJECT_ATTRIBUTE_SAM_PROBE_JOB,
    )

    sam_jobs = {SUBJECT_ATTRIBUTE_SAM_PROBE_JOB, SUBJECT_ATTRIBUTE_COMPLETION_SAM_JOB}
    binary_jobs = sam_jobs | {SUBJECT_ATTRIBUTE_COMPLETION_GENERATE_JOB}
    indexed: dict[tuple[str, str], BinaryJobIds] = {}
    job_owners: dict[str, tuple[str, str]] = {}
    phases = Path(group_root) / "phases"
    if phases.is_symlink():
        raise ValueError(f"symlinked phases directory is unsafe: {phases}")
    if not phases.is_dir():
        return indexed
    for phase in sorted(phases.iterdir()):
        if phase.is_symlink():
            raise ValueError(f"symlinked phase directory is unsafe: {phase}")
        if not phase.is_dir():
            continue
        plan_path = phase / "plan.json"
        if not plan_path.exists():
            continue
        payload = _read_object(plan_path)
        jobs = payload.get("jobs")
        if (
            not isinstance(jobs, list)
            or type(payload.get("job_count")) is not int
            or payload["job_count"] != len(jobs)
            or not isinstance(payload.get("plan_hash"), str)
        ):
            raise ValueError(f"invalid phase plan structure: {plan_path}")
        for record in jobs:
            if not isinstance(record, dict):
                raise TypeError(f"invalid phase plan job record: {plan_path}")
            job_type = record.get("job_type")
            if job_type not in binary_jobs:
                continue
            job_id = record.get("job_id")
            shard = record.get("canonical_shard")
            if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
                raise ValueError(f"unsafe phase plan job_id: {job_id!r}")
            if not isinstance(shard, str) or _SHARD_NAME.fullmatch(shard) is None:
                raise ValueError(f"invalid phase plan canonical_shard: {shard!r}")
            clip_uid = _safe_clip_uid(record.get("clip_uid"))
            key = (shard, clip_uid)
            previous = job_owners.setdefault(job_id, key)
            if previous != key:
                raise ValueError(f"binary job {job_id} belongs to multiple clips")
            jobs_for_clip = indexed.setdefault(key, BinaryJobIds())
            if job_type in sam_jobs:
                jobs_for_clip.sam.add(job_id)
            else:
                jobs_for_clip.completion.add(job_id)
    return indexed


def _safe_binary_branch(group_root: Path, kind: str) -> Path:
    root = Path(group_root)
    if root.is_symlink():
        raise ValueError(f"symlinked group directory is unsafe: {root}")
    parts = ("semantic", "subject_attributes", "binary", kind)
    cursor = root
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"symlinked Subject Attribute binary path is unsafe: {cursor}")
    return cursor


def _clean_job_dir(
    directory: Path,
    *,
    pattern: str,
    file_counter: str,
    directory_counter: str,
    apply: bool,
    stats: dict[str, int],
) -> None:
    if directory.is_symlink():
        stats["failures"] += 1
        return
    if not directory.is_dir():
        stats["already_missing"] += 1
        return
    try:
        files = sorted(directory.glob(pattern))
        if any(path.is_symlink() or not path.is_file() for path in files):
            stats["failures"] += 1
            return
        if not files:
            stats["already_missing"] += 1
        for path in files:
            if apply:
                try:
                    path.unlink()
                except OSError:
                    stats["failures"] += 1
                    continue
            stats[file_counter] += 1
        if not any(directory.iterdir()):
            if apply:
                directory.rmdir()
            stats[directory_counter] += 1
        elif not apply and files and len(files) == len(tuple(directory.iterdir())):
            # In preview, those exact files are all that would remain to remove.
            stats[directory_counter] += 1
    except OSError:
        stats["failures"] += 1


def cleanup_clip_binary(
    group_root: Path, jobs: BinaryJobIds, *, apply: bool
) -> dict[str, int]:
    """Clean only indexed job directories after the caller proves terminality.

    A failed unlink cannot undo an already-published terminal outcome. The
    returned failure count is execution-only telemetry for retrying GC later.
    """
    stats = _stats()
    try:
        sam_root = _safe_binary_branch(group_root, "sam")
        completion_root = _safe_binary_branch(group_root, "completion")
    except ValueError:
        stats["failures"] += 1
        return stats
    for job_id in sorted(jobs.sam):
        if _JOB_ID.fullmatch(job_id) is None:
            stats["failures"] += 1
            continue
        _clean_job_dir(
            sam_root / job_id,
            pattern="mask-*.npy",
            file_counter="sam_files_cleaned",
            directory_counter="sam_job_dirs_cleaned",
            apply=apply,
            stats=stats,
        )
    for job_id in sorted(jobs.completion):
        if _JOB_ID.fullmatch(job_id) is None:
            stats["failures"] += 1
            continue
        _clean_job_dir(
            completion_root / job_id,
            pattern="generated.png",
            file_counter="completion_pngs_cleaned",
            directory_counter="completion_job_dirs_cleaned",
            apply=apply,
            stats=stats,
        )
    return stats


def _group_shards(group_root: Path) -> tuple[str, ...]:
    payload = _read_object(group_root / "group.json")
    shards = payload.get("canonical_shards")
    if (
        payload.get("group_id") != group_root.name
        or not isinstance(payload.get("group_identity"), str)
        or not isinstance(shards, list)
        or not shards
        or any(
            not isinstance(shard, str) or _SHARD_NAME.fullmatch(shard) is None
            for shard in shards
        )
        or len(shards) != len(set(shards))
    ):
        raise ValueError(f"invalid resource epoch group descriptor: {group_root}")
    return tuple(shards)


def _terminal_keys(group_root: Path, shards: tuple[str, ...]) -> tuple[set[tuple[str, str]], int]:
    from r2v_data_v2.v3.post_mask_epoch_subject_attributes import (
        SUBJECT_ATTRIBUTE_CLIP_OUTCOME_SCHEMA,
        SUBJECT_ATTRIBUTE_CLIP_PLAN_SCHEMA,
    )
    from r2v_data_v2.v3.subject_attributes import ClipEnrichmentResult, EnrichmentTotals

    empty_counts = ClipEnrichmentResult(
        clip_uid="", totals=EnrichmentTotals(), owner_limit_reached=False,
        enriched_sample=None,
    ).to_counts()
    semantic = group_root / "semantic" / "subject_attributes"
    for path in (group_root / "semantic", semantic, semantic / "plans", semantic / "outcomes"):
        if path.is_symlink():
            raise ValueError(f"symlinked Subject Attribute authority is unsafe: {path}")
    terminals: set[tuple[str, str]] = set()
    failures = 0
    for shard in shards:
        directory = semantic / "outcomes" / shard
        plan_directory = semantic / "plans" / shard
        if plan_directory.is_symlink():
            raise ValueError(f"symlinked clip plans directory is unsafe: {plan_directory}")
        if directory.is_symlink():
            raise ValueError(f"symlinked clip outcomes directory is unsafe: {directory}")
        if not directory.is_dir():
            continue
        for outcome_path in sorted(directory.glob("*.json")):
            try:
                outcome = _read_object(outcome_path)
                clip_uid = _safe_clip_uid(outcome_path.stem)
                if outcome.get("terminal") != "ready":
                    continue
                plan = _read_object(plan_directory / f"{clip_uid}.json")
                owners = plan.get("subject_owners")
                counts = outcome.get("counts")
                if (
                    set(outcome)
                    != {
                        "schema", "clip_uid", "terminal", "counts",
                        "owner_outcome_ids", "enriched_sample_path",
                        "enriched_sample_sha256",
                    }
                    or set(plan)
                    != {
                        "schema", "clip_uid", "policy", "clip_digest",
                        "effective_export", "frames_sha256", "masks_sha256",
                        "classification", "subject_owners",
                    }
                    or outcome.get("schema") != SUBJECT_ATTRIBUTE_CLIP_OUTCOME_SCHEMA
                    or outcome.get("clip_uid") != clip_uid
                    or not isinstance(counts, dict)
                    or set(counts) != set(empty_counts)
                    or any(
                        type(value) is not type(empty_counts[name])
                        or not math.isfinite(value)
                        or value < 0
                        for name, value in counts.items()
                    )
                    or plan.get("schema") != SUBJECT_ATTRIBUTE_CLIP_PLAN_SCHEMA
                    or plan.get("clip_uid") != clip_uid
                    or not isinstance(plan.get("policy"), dict)
                    or not isinstance(plan.get("clip_digest"), str)
                    or not isinstance(plan.get("effective_export"), dict)
                    or plan.get("classification") not in {"no_work", "owners"}
                    or not isinstance(owners, list)
                    or any(
                        not isinstance(owner, dict)
                        or not isinstance(owner.get("owner_entity_id"), str)
                        or not isinstance(owner.get("eligible"), bool)
                        for owner in owners
                    )
                    or (plan.get("classification") == "no_work" and bool(owners))
                    or (plan.get("classification") == "no_work" and counts != empty_counts)
                    or outcome.get("owner_outcome_ids")
                    != [owner["owner_entity_id"] for owner in owners if owner["eligible"]]
                    or outcome.get("enriched_sample_path") is not None
                    and not isinstance(outcome["enriched_sample_path"], str)
                    or outcome.get("enriched_sample_sha256") is not None
                    and not isinstance(outcome["enriched_sample_sha256"], str)
                    or (outcome.get("enriched_sample_path") is None)
                    != (outcome.get("enriched_sample_sha256") is None)
                ):
                    raise ValueError(f"invalid ready Subject Attribute outcome: {outcome_path}")
                terminals.add((shard, clip_uid))
            except (OSError, TypeError, ValueError):
                failures += 1
    return terminals, failures


def _gc_group(group_root: Path, *, apply: bool) -> dict[str, int]:
    stats = _stats()
    stats["groups_seen"] = 1
    try:
        shards = _group_shards(group_root)
        indexed = load_binary_job_index(group_root)
        terminals, failures = _terminal_keys(group_root, shards)
    except (OSError, TypeError, ValueError):
        stats["failures"] += 1
        return stats
    stats["terminal_clips_seen"] = len(terminals)
    stats["failures"] += failures
    for key in sorted(terminals):
        jobs = indexed.get(key)
        if jobs is None:
            continue
        cleaned = cleanup_clip_binary(group_root, jobs, apply=apply)
        for name in _STATS:
            stats[name] += cleaned[name]
    return stats


def gc_state_root(state_root: Path, *, apply: bool) -> dict[str, int]:
    """One explicit backfill pass over existing groups, never over media trees.

    ``state_root`` is the existing ``.../state/resource_epochs`` directory.
    Applying takes the same nonblocking group flock as formal production;
    preview never creates a lock file or any other state.
    """
    state_root = Path(state_root)
    if state_root.is_symlink() or not state_root.is_dir():
        raise ValueError(f"resource epoch state root is invalid: {state_root}")
    total = _stats()
    for group_root in sorted(state_root.iterdir()):
        if _GROUP_NAME.fullmatch(group_root.name) is None:
            continue
        if group_root.is_symlink() or not group_root.is_dir():
            total["groups_seen"] += 1
            total["failures"] += 1
            continue
        if apply:
            lock_path = state_root / f"{group_root.name}.lock"
            if lock_path.is_symlink():
                group_stats = _stats()
                group_stats["groups_seen"] = 1
                group_stats["failures"] = 1
            else:
                with file_lock(lock_path, blocking=False) as held:
                    group_stats = (
                        _gc_group(group_root, apply=True)
                        if held
                        else {**_stats(), "groups_seen": 1, "failures": 1}
                    )
        else:
            group_stats = _gc_group(group_root, apply=False)
        for name in _STATS:
            total[name] += group_stats[name]
    return total
