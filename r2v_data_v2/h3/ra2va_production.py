"""Durable shard scheduling for the existing RA2VA reconcile contract."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from pathlib import Path

from r2v_data_v2.h3.t2va_production import (
    SHARD_SIZE,
    EndpointUnavailable,
    _endpoint_error,
    _repair_tail,
    append_row,
    atomic_json,
    file_lock,
    load_states,
    shard_name,
    shard_order,
)


def run_reconcile_clip(
    *, job, stem_record, backend, destination: Path, route: str,
    allow_unverified: bool,
) -> dict:
    """Use the existing reconcile publisher and validators for one clip."""
    from r2v_data_v2.h3.mimo25_stem_shadow import run_mimo25_stem_reconcile_shadow

    run_mimo25_stem_reconcile_shadow(
        jobs=[job], stem_records=[stem_record], backend=backend,
        output_root=destination, source_clip_uids=[job.clip_uid],
        route=route, binding_evidence_mode=job.binding_evidence_mode,
        allow_unverified=allow_unverified,
    )
    with (destination / "records.jsonl").open(encoding="utf-8") as handle:
        return json.loads(handle.readline())


def _ready_record(clip_root: Path) -> dict | None:
    path = clip_root / "reconcile" / "records.jsonl"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    if "schema_version" in record:
        from r2v_data_v2.h3.mimo25_stem_shadow import MimoStemReconcileRecord

        MimoStemReconcileRecord.model_validate(record)
    return record if record.get("status") == "ready" else None


def ready_reconcile_records(root: Path, shard_id: int) -> list[dict]:
    shard = root / "shards" / shard_name(shard_id)
    source = json.loads((shard / "sources.json").read_text(encoding="utf-8"))
    records = []
    for uid in source:
        record = _ready_record(shard / "artifacts" / uid)
        if record is not None:
            if record.get("clip_uid") != uid:
                raise ValueError("RA2VA ready artifact differs from shard source clip")
            records.append(record)
    return records


def publish_ready_reconcile_view(root: Path, destination: Path, shard_ids: list[int]) -> Path:
    """Expose one frozen MiMo record set to existing R2VA/RA2VA downstream tools."""
    if len(set(shard_ids)) != len(shard_ids):
        raise ValueError("duplicate RA2VA shard in ready view")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ra2va-ready-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "view"
        stage.mkdir()
        seen = set()
        with (stage / "records.jsonl").open("w", encoding="utf-8") as handle:
            for shard_id in sorted(shard_ids):
                for record in ready_reconcile_records(root, shard_id):
                    uid = record["clip_uid"]
                    if uid in seen:
                        raise ValueError("duplicate ready RA2VA clip across shards")
                    seen.add(uid)
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(destination)
        os.replace(stage, destination)
    return destination


def process_reconcile_shard(
    *, root: Path, shard_id: int, clip_uids: list[str],
    run_clip: Callable[[str, Path], dict],
    skipped: dict[str, str] | None = None,
    request_workers: int = 2,
    stage: Callable[[int], object] | None = None,
) -> dict[str, dict]:
    """Publish each ready clip before journaling; retry failed clips next invocation."""
    if request_workers < 1 or len(clip_uids) > SHARD_SIZE or len(set(clip_uids)) != len(clip_uids):
        raise ValueError("invalid RA2VA shard inventory or worker count")
    skipped = skipped or {}
    shard = root / "shards" / shard_name(shard_id)
    with file_lock(shard / "shard.lock"):
        source_path = shard / "sources.json"
        if source_path.exists():
            if json.loads(source_path.read_text(encoding="utf-8")) != clip_uids:
                raise ValueError("RA2VA shard source order changed")
        else:
            atomic_json(source_path, clip_uids)
        journal = shard / "state.jsonl.partial"
        _repair_tail(journal)
        states = load_states(shard)
        if (shard / "COMPLETE").exists():
            return states

        pending = []
        for index, uid in enumerate(clip_uids):
            clip_root = shard / "artifacts" / uid
            ready = _ready_record(clip_root)
            if ready is not None and ready.get("clip_uid") != uid:
                raise ValueError("RA2VA ready artifact differs from shard source clip")
            if ready is not None:
                state = {"source_index": shard_id * SHARD_SIZE + index,
                         "clip_uid": uid, "status": "ready", "failure_reason": None}
                if states.get(uid) != state:
                    append_row(journal, state)
                    states[uid] = state
            elif states.get(uid, {}).get("status") == "skipped":
                continue
            elif uid in skipped:
                state = {"source_index": shard_id * SHARD_SIZE + index,
                         "clip_uid": uid, "status": "skipped", "failure_reason": skipped[uid]}
                append_row(journal, state)
                states[uid] = state
            else:
                pending.append((index, uid))

        def attempt(index: int, uid: str) -> dict:
            clip_root = shard / "artifacts" / uid
            clip_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".attempt-", dir=clip_root) as temporary:
                candidate = Path(temporary) / "reconcile"
                record = run_clip(uid, candidate)
                _endpoint_error(record.get("failure_reason"))
                if record.get("status") == "ready":
                    os.replace(candidate, clip_root / "reconcile")
                elif record.get("status") == "failed":
                    atomic_json(clip_root / "latest_failed.json", record)
                else:
                    raise ValueError("RA2VA reconcile returned an unknown clip status")
                return {
                    "source_index": shard_id * SHARD_SIZE + index,
                    "clip_uid": uid,
                    "status": record["status"],
                    "failure_reason": record.get("failure_reason"),
                }

        if pending:
            @contextmanager
            def endpoint_stage():
                try:
                    with stage(shard_id) if stage is not None else nullcontext():
                        yield
                except (ConnectionError, OSError) as exc:
                    raise EndpointUnavailable(str(exc)) from exc

            context = endpoint_stage()
            with context, ThreadPoolExecutor(max_workers=request_workers) as pool:
                source = iter(pending)
                futures = {}

                def fill() -> None:
                    while len(futures) < request_workers:
                        try:
                            index, uid = next(source)
                        except StopIteration:
                            break
                        futures[pool.submit(attempt, index, uid)] = (index, uid)

                fill()
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        index, uid = futures.pop(future)
                        try:
                            state = future.result()
                        except EndpointUnavailable:
                            for other in futures:
                                other.cancel()
                            raise
                        except Exception as exc:  # noqa: BLE001 - one semantic failure affects one clip
                            _endpoint_error(exc)
                            state = {
                                "source_index": shard_id * SHARD_SIZE + index,
                                "clip_uid": uid, "status": "failed",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                            }
                        append_row(journal, state)
                        states[state["clip_uid"]] = state
                    fill()
        if all(states.get(uid, {}).get("status") in {"ready", "skipped"} for uid in clip_uids):
            atomic_json(shard / "COMPLETE", {"clip_count": len(clip_uids)})
        return states


__all__ = [
    "process_reconcile_shard", "publish_ready_reconcile_view", "ready_reconcile_records",
    "run_reconcile_clip", "shard_order",
]
