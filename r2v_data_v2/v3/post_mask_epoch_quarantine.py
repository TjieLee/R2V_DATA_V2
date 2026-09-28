"""Invocation-local isolation of damaged Post-Mask clip artifacts.

Call ``record_if_local`` only around a known current-clip artifact read. Plan,
ledger, resource, and model operations are never candidate quarantine sites.
"""

from __future__ import annotations

import errno
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import ValidationError

from r2v_data_v2.v3.post_mask_epoch_state import atomic_write_json

_MARKER_SCHEMA = "post_mask_clip_quarantine/1"


def _local_artifact_error(
    exc: Exception,
    storage: Any,
    clip_uid: str,
    *,
    known_clip_artifact_read: bool = False,
) -> bool:
    if isinstance(exc, PermissionError):
        return False
    if known_clip_artifact_read and isinstance(
        exc, (ValidationError, json.JSONDecodeError, ValueError)
    ):
        return True
    if not isinstance(exc, OSError):
        return False
    if exc.errno not in (None, errno.ENOENT, errno.ENOTDIR, errno.EISDIR):
        return False
    filename = exc.filename
    if filename is None:
        # RunStorage._require_clip raises without a filename. It is safe to
        # recognize only its exact clip-specific message, not arbitrary I/O.
        return isinstance(exc, FileNotFoundError) and str(exc) == (
            f"clip.json does not exist for {clip_uid}"
        )
    clip_dir = Path(os.path.abspath(storage.clip_dir(clip_uid)))
    artifact = Path(os.path.abspath(filename))
    return artifact == clip_dir or clip_dir in artifact.parents


class ClipQuarantine:
    """One failure record per clip, with optional group-local restart state."""

    def __init__(
        self,
        emit: Any = None,
        *,
        marker_path: Path | None = None,
        eligible: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._emit = emit
        self._marker_path = marker_path
        self.eligible = {
            str(shard): tuple(str(uid) for uid in uids)
            for shard, uids in (eligible or {}).items()
        }
        self._failures: dict[tuple[str, str], tuple[str, str]] = {}
        self._lock = Lock()

    @classmethod
    def for_group(
        cls,
        ledger_root: Path,
        hydrated: Mapping[str, Sequence[str]],
        *,
        emit: Any = None,
        storages: Mapping[str, Any] | None = None,
    ) -> ClipQuarantine:
        """Restore only recorded local exclusions, never a changed healthy scope."""
        path = Path(ledger_root) / "composition" / "clip_quarantine.json"
        current = {
            str(shard): tuple(str(uid) for uid in uids)
            for shard, uids in hydrated.items()
        }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(emit, marker_path=path, eligible=current)
        if not isinstance(payload, dict) or set(payload) != {
            "schema", "eligible_clip_uids_by_shard", "quarantined"
        } or payload["schema"] != _MARKER_SCHEMA:
            raise ValueError(f"invalid clip quarantine marker: {path}")
        raw_eligible = payload["eligible_clip_uids_by_shard"]
        raw_quarantined = payload["quarantined"]
        if not isinstance(raw_eligible, dict) or not isinstance(raw_quarantined, list):
            raise TypeError(f"invalid clip quarantine marker: {path}")
        if set(raw_eligible) != set(current) or any(
            not isinstance(shard, str)
            or not isinstance(uids, list)
            or any(not isinstance(uid, str) or not uid for uid in uids)
            or len(uids) != len(set(uids))
            for shard, uids in raw_eligible.items()
        ):
            raise ValueError(f"clip quarantine eligible scope drifted: {path}")
        original = {shard: tuple(uids) for shard, uids in raw_eligible.items()}
        instance = cls(emit, marker_path=path, eligible=original)
        for item in raw_quarantined:
            if not isinstance(item, dict) or set(item) != {
                "shard", "clip_uid", "stage", "reason"
            }:
                raise ValueError(f"invalid clip quarantine marker: {path}")
            shard, uid = item["shard"], item["clip_uid"]
            stage, reason = item["stage"], item["reason"]
            if (
                not isinstance(shard, str)
                or not isinstance(uid, str)
                or not isinstance(stage, str)
                or not stage
                or not isinstance(reason, str)
                or not reason
                or uid not in original.get(shard, ())
                or (shard, uid) in instance._failures
            ):
                raise ValueError(f"invalid clip quarantine marker: {path}")
            instance._failures[(shard, uid)] = (stage, reason)
        for shard, uids in original.items():
            missing = set(uids) - set(current[shard])
            if missing - {uid for saved_shard, uid in instance._failures if saved_shard == shard}:
                raise ValueError(f"clip quarantine eligible scope drifted: {path}")
            if [uid for uid in uids if uid in current[shard]] != list(current[shard]):
                raise ValueError(f"clip quarantine eligible scope drifted: {path}")
        if storages is None or set(storages) != set(original):
            raise ValueError(f"clip quarantine has no failure authority: {path}")
        pending = {
            (shard, uid, stage, reason)
            for (shard, uid), (stage, reason) in instance._failures.items()
        }
        for shard in sorted({item[0] for item in pending}):
            failure_path = Path(storages[shard].root) / "failures.jsonl"
            try:
                with failure_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        if record.get("details", {}).get("post_mask_quarantine") is True:
                            pending.discard((
                                shard,
                                record.get("clip_uid"),
                                record.get("stage"),
                                record.get("reason"),
                            ))
            except (OSError, ValueError, AttributeError) as exc:
                raise ValueError(
                    f"clip quarantine failure record is unreadable: {failure_path}"
                ) from exc
        if pending:
            raise ValueError(f"clip quarantine without failure record: {path}")
        return instance

    def excluded_by_shard(self) -> dict[str, list[str]]:
        with self._lock:
            return {
                shard: [uid for uid in uids if (shard, uid) in self._failures]
                for shard, uids in sorted(self.eligible.items())
                if any((shard, uid) in self._failures for uid in uids)
            }

    def _write_marker(
        self, failures: dict[tuple[str, str], tuple[str, str]]
    ) -> None:
        if self._marker_path is None:
            return
        atomic_write_json(
            self._marker_path,
            {
                "schema": _MARKER_SCHEMA,
                "eligible_clip_uids_by_shard": {
                    shard: list(uids) for shard, uids in sorted(self.eligible.items())
                },
                "quarantined": [
                    {"shard": shard, "clip_uid": uid, "stage": stage, "reason": reason}
                    for (shard, uid), (stage, reason) in sorted(failures.items())
                ],
            },
        )

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._failures)

    def contains(self, shard: str, clip_uid: str) -> bool:
        with self._lock:
            return (shard, clip_uid) in self._failures

    def record_if_local(
        self,
        shard: str,
        storage: Any,
        clip_uid: str,
        stage: str,
        exc: Exception,
        *,
        known_clip_artifact_read: bool = False,
    ) -> bool:
        """Return False for a non-local error so the caller can re-raise it."""
        if not _local_artifact_error(
            exc,
            storage,
            clip_uid,
            known_clip_artifact_read=known_clip_artifact_read,
        ):
            return False
        key = (shard, clip_uid)
        with self._lock:
            if key in self._failures:
                return True
            if self._marker_path is not None and clip_uid not in self.eligible.get(shard, ()):
                raise ValueError("cannot quarantine clip outside hydrated group scope")
            reason = f"{type(exc).__name__}: {exc}"
            storage.append_failure(
                stage=stage,
                clip_uid=clip_uid,
                reason=reason,
                details={"post_mask_quarantine": True},
            )
            self._write_marker({**self._failures, key: (stage, reason)})
            self._failures[key] = (stage, reason)
            if self._emit is not None:
                self._emit(
                    "post_mask_clip_quarantined",
                    stage=stage,
                    shard=shard,
                    clip_uid=clip_uid,
                    reason=reason,
                )
        return True
