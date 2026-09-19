"""8-shard resource-epoch group identity, static ownership and group locking.

A resource epoch group is an *internal execution concept only*. It never
appears in the public dataset schema, in a shard semantic identity, or in any
exported artifact.

Group identity is derived from exactly three things:

    ordered canonical shard identities
    campaign semantic identity
    group size + group schema version

It therefore excludes hostname, ``RANK``, ``WORLD_SIZE``, physical GPU id, PID
and runtime concurrency. That exclusion is what lets a campaign be restarted
under a different topology and still recognise the same group, its plan and its
receipts.

Multi-node ownership is static: node ``RANK`` owns the groups satisfying
``group_index % WORLD_SIZE == RANK``. There is no work stealing, no broker and
no cross-node queue. A group has one shared advisory lock; a node that cannot
take it leaves that group alone.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from r2v_data_v2.v3.post_mask_epoch_jobs import canonical_json
from r2v_data_v2.v3.post_mask_epoch_state import (
    _fsync_directory,
    atomic_write_json,
    file_lock,
)

#: Default number of canonical shards per resource epoch group.
DEFAULT_GROUP_SIZE = 8

GROUP_SCHEMA_VERSION = "post_mask_resource_epoch_v3/group/1"

#: Group outcome markers. Historical diagnostics are appended, never rewritten.
GROUP_INCOMPLETE = "incomplete"
GROUP_COMPLETED = "completed"


def campaign_identity(semantic: Mapping[str, Any]) -> str:
    """Digest the campaign's semantic identity (config/dataset), nothing else."""
    return hashlib.sha256(canonical_json(semantic).encode("utf-8")).hexdigest()


@dataclass(frozen=True, order=True)
class ResourceEpochGroup:
    group_index: int
    group_id: str
    canonical_shards: tuple[str, ...]
    campaign_identity: str
    group_size: int = DEFAULT_GROUP_SIZE
    schema_version: str = GROUP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if len(self.canonical_shards) > self.group_size:
            raise ValueError("group holds more shards than group_size")

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "group_size": self.group_size,
            "campaign_identity": self.campaign_identity,
            "canonical_shards": list(self.canonical_shards),
        }

    def identity(self) -> str:
        """Execution-free identity: no rank, world size, host, GPU or PID."""
        return hashlib.sha256(
            canonical_json(self.identity_payload()).encode("utf-8")
        ).hexdigest()


def group_id_for(index: int) -> str:
    return f"group-{index:06d}"


def build_groups(
    canonical_shards: Sequence[str],
    *,
    campaign: Mapping[str, Any],
    group_size: int = DEFAULT_GROUP_SIZE,
) -> tuple[ResourceEpochGroup, ...]:
    """Chunk lexically sorted canonical shard identities into fixed groups."""
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    ordered = tuple(sorted(str(shard) for shard in canonical_shards))
    if len(ordered) != len(set(ordered)):
        raise ValueError("duplicate canonical shard identity")
    campaign_digest = campaign_identity(campaign)
    groups = []
    for index, start in enumerate(range(0, len(ordered), group_size)):
        chunk = ordered[start : start + group_size]
        groups.append(
            ResourceEpochGroup(
                group_index=index,
                group_id=group_id_for(index),
                canonical_shards=chunk,
                campaign_identity=campaign_digest,
                group_size=group_size,
            )
        )
    return tuple(groups)


def assigned_groups(
    groups: Sequence[ResourceEpochGroup],
    *,
    rank: int,
    world_size: int,
) -> tuple[ResourceEpochGroup, ...]:
    """Static ownership: ``group_index % world_size == rank``. No stealing."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must satisfy 0 <= rank < world_size")
    return tuple(
        group for group in groups if group.group_index % world_size == rank
    )


def resource_epoch_root(post_mask_state_root: Path) -> Path:
    return Path(post_mask_state_root) / "resource_epochs"


def group_root(post_mask_state_root: Path, group: ResourceEpochGroup) -> Path:
    return resource_epoch_root(post_mask_state_root) / group.group_id


def group_lock_path(post_mask_state_root: Path, group: ResourceEpochGroup) -> Path:
    return resource_epoch_root(post_mask_state_root) / f"{group.group_id}.lock"


def group_descriptor_path(post_mask_state_root: Path, group: ResourceEpochGroup) -> Path:
    return group_root(post_mask_state_root, group) / "group.json"


def write_group_descriptor(
    post_mask_state_root: Path, group: ResourceEpochGroup
) -> Path:
    """Publish the immutable group descriptor, validating any existing copy."""
    payload = {
        **group.identity_payload(),
        "group_id": group.group_id,
        "group_index": group.group_index,
        "group_identity": group.identity(),
    }
    path = group_descriptor_path(post_mask_state_root, group)
    if path.is_file():
        if json.loads(path.read_text()) != payload:
            raise ValueError(
                f"resource epoch group identity mismatch at {path}"
            )
        return path
    atomic_write_json(path, payload)
    return path


@contextmanager
def group_ownership(
    post_mask_state_root: Path, group: ResourceEpochGroup
) -> Iterator[bool]:
    """Take the group's shared lock. Yields False when another node owns it."""
    with file_lock(group_lock_path(post_mask_state_root, group), blocking=False) as held:
        yield held


def record_group_outcome(
    post_mask_state_root: Path,
    group: ResourceEpochGroup,
    *,
    outcome: str,
    reason: str = "",
) -> Path:
    """Append a group outcome marker. History is never deleted."""
    root = group_root(post_mask_state_root, group)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "outcomes.jsonl"
    payload = canonical_json(
        {
            "group_id": group.group_id,
            "group_identity": group.identity(),
            "outcome": outcome,
            "reason": reason,
        }
    )
    was_new = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")
        handle.flush()
        # Shared filesystems make an un-fsynced append lose the outcome when
        # the node dies; history stays append-only either way.
        os.fsync(handle.fileno())
    if was_new:
        # A newly created outcomes file needs its directory entry durable too.
        _fsync_directory(path.parent)
    return path


def read_group_outcomes(
    post_mask_state_root: Path, group: ResourceEpochGroup
) -> list[dict[str, Any]]:
    path = group_root(post_mask_state_root, group) / "outcomes.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
