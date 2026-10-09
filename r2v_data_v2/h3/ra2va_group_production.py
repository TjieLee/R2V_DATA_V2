"""CPU pilot coordinator: shared groups, atomic claims, and terminal results."""

from __future__ import annotations

import fcntl
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from r2v_data_v2.h3.ra2va_group_source import (
    GroupTask,
    discover_published_groups,
    freeze_pilot_inventory,
    read_inventory_task,
)
from r2v_data_v2.h3.t2va_production import atomic_json, file_lock

STAGES = ("canonical", "sam", "auk", "resolve", "diarizen", "asr", "mimo", "export")
TERMINAL = ("ready", "failed", "skipped")


def read_json(path):
    return json.loads(path.read_text())


def try_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


@dataclass(frozen=True)
class StageSnapshot:
    group_id: str
    stage: str
    generation: int
    phase: str


@dataclass
class TaskClaim:
    snapshot: StageSnapshot
    task: GroupTask
    worker_id: str
    lock_handle: object

    def close(self):
        self.lock_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class StaleGeneration(RuntimeError):
    pass


class GroupCoordinator:
    def __init__(self, run_root: Path, source_root: Path, limit: int, *, max_groups=1):
        if not 1 <= limit <= 200 or max_groups < 1:
            raise ValueError("invalid bounded pilot settings")
        self.run_root = run_root.absolute()
        self.source_root = source_root.absolute()
        self.limit = limit
        self.max_groups = max_groups
        self.control_path = self.run_root / "control.json"
        settings = {"source_root": str(self.source_root), "limit": limit, "max_groups": max_groups}
        with file_lock(self.run_root / "control.lock", blocking=True):
            if self.control_path.exists():
                if read_json(self.control_path)["settings"] != settings:
                    raise ValueError("resume settings differ from persisted run")
            else:
                atomic_json(self.control_path, {
                    "mode": "cpu_fake_pilot", "settings": settings, "generation": 0,
                    "group_id": None, "stage": None, "phase": "idle", "started_groups": [],
                })

    def group_root(self, group_id):
        return self.run_root / "groups" / group_id

    def stage_root(self, snapshot):
        return self.group_root(snapshot.group_id) / "stages" / f"{snapshot.generation:06d}-{snapshot.stage}"

    def result_path(self, snapshot, ordinal):
        return self.stage_root(snapshot) / "results" / f"{ordinal:06d}.json"

    @staticmethod
    def _snapshot(control):
        return StageSnapshot(*(control[k] for k in ("group_id", "stage", "generation", "phase")))

    def _initialize_stage(self, snapshot):
        path = self.stage_root(snapshot) / "dispatch.json"
        if not path.exists():
            atomic_json(path, {"generation": snapshot.generation, "cursor": 0, "active": {},
                               "ready": 0, "failed": 0, "skipped": 0, "started_at": time.time()})

    def select_current(self):
        with file_lock(self.run_root / "control.lock", blocking=True):
            control = read_json(self.control_path)
            if control["group_id"] and control["phase"] != "complete":
                return self._snapshot(control)
            if len(control["started_groups"]) >= self.max_groups:
                return None
            # An inventory published before a control-write crash is already started.
            roots = self.run_root / "groups"
            unfinished = sorted(p.parent.name for p in roots.glob("*/inventory/inventory.json")
                                if not (p.parent.parent / "PILOT_COMPLETE").exists())
            if unfinished:
                group_id = unfinished[0]
            else:
                candidates = [g for g in discover_published_groups(self.source_root)
                              if g.group_id not in control["started_groups"]]
                if not candidates:
                    return None
                group = candidates[0]
                group_id = group.group_id
                freeze_pilot_inventory(group, self.group_root(group_id) / "inventory", self.limit)
            control.update(group_id=group_id, stage=STAGES[0],
                           generation=control["generation"] + 1, phase="running", members={})
            if group_id not in control["started_groups"]:
                control["started_groups"].append(group_id)
            snapshot = self._snapshot(control)
            self._initialize_stage(snapshot)
            atomic_json(self.control_path, control)
            return snapshot

    @contextmanager
    def _checked(self, snapshot):
        # All metadata writers take control -> dispatch. Clip acquisition is NEVER blocking.
        with file_lock(self.run_root / "control.lock", blocking=True):
            control = read_json(self.control_path)
            if any(control[k] != getattr(snapshot, k) for k in ("generation", "group_id", "stage")):
                raise StaleGeneration("worker belongs to an earlier stage generation")
            with file_lock(self.stage_root(snapshot) / "dispatch.lock", blocking=True):
                yield control, read_json(self.stage_root(snapshot) / "dispatch.json")

    def _save_dispatch(self, snapshot, dispatch):
        atomic_json(self.stage_root(snapshot) / "dispatch.json", dispatch)

    def _account_results(self, snapshot, dispatch):
        changed = False
        for key in list(dispatch["active"]):
            path = self.result_path(snapshot, int(key))
            if path.exists():
                result = read_json(path)
                if result["generation"] != snapshot.generation:
                    raise StaleGeneration("result generation differs")
                dispatch[result["status"]] += 1
                del dispatch["active"][key]
                changed = True
        if changed:
            self._save_dispatch(snapshot, dispatch)

    def claim(self, snapshot, worker_id):
        with self._checked(snapshot) as (control, dispatch):
            if control["phase"] != "running":
                return None
            self._account_results(snapshot, dispatch)
            inventory = self.group_root(snapshot.group_id) / "inventory"
            count = read_json(inventory / "inventory.json")["selected_count"]
            candidates = [int(key) for key in dispatch["active"]]
            if dispatch["cursor"] < count:
                candidates.append(dispatch["cursor"])
            for ordinal in candidates:
                handle = try_lock(self.stage_root(snapshot) / "clips" / f"{ordinal:06d}.lock")
                if handle is None:
                    continue
                try:
                    task = read_inventory_task(inventory, ordinal)
                    key = str(ordinal)
                    recovered = key in dispatch["active"]
                    dispatch["active"][key] = {"worker_id": worker_id, "generation": snapshot.generation}
                    if not recovered:
                        dispatch["cursor"] += 1
                    self._save_dispatch(snapshot, dispatch)
                    self.log("reclaimed" if recovered else "claimed", snapshot,
                             worker_id=worker_id, clip_uid=task.clip_uid, ordinal=ordinal)
                    return TaskClaim(snapshot, task, worker_id, handle)
                except BaseException:
                    handle.close()
                    raise
            return None

    def publish(self, claim, status, payload, elapsed_seconds):
        if status not in TERMINAL or claim.lock_handle.closed:
            raise ValueError("publication requires terminal status and owned clip lock")
        snapshot = claim.snapshot
        with self._checked(snapshot) as (_, dispatch):
            key = str(claim.task.ordinal)
            path = self.result_path(snapshot, claim.task.ordinal)
            if path.exists():
                self._account_results(snapshot, dispatch)
                return
            if dispatch["active"].get(key, {}).get("worker_id") != claim.worker_id:
                raise ValueError("claim owner differs")
            atomic_json(path, {"generation": snapshot.generation, "group_id": snapshot.group_id,
                               "stage": snapshot.stage, "ordinal": claim.task.ordinal,
                               "clip_uid": claim.task.clip_uid, "status": status,
                               "worker_id": claim.worker_id, "elapsed_seconds": elapsed_seconds,
                               "payload": payload})
            self._account_results(snapshot, dispatch)
            self.log("published", snapshot, clip_uid=claim.task.clip_uid, status=status,
                     worker_id=claim.worker_id, elapsed_seconds=elapsed_seconds)

    def recover(self, snapshot):
        with self._checked(snapshot) as (_, dispatch):
            self._account_results(snapshot, dispatch)

    def progress(self, snapshot):
        with self._checked(snapshot) as (control, dispatch):
            self._account_results(snapshot, dispatch)
            total = read_json(self.group_root(snapshot.group_id) / "inventory/inventory.json")["selected_count"]
            done = sum(dispatch[k] for k in TERMINAL)
            elapsed = max(time.time() - dispatch["started_at"], 0.000001)
            return {"group": snapshot.group_id, "stage": snapshot.stage,
                    "generation": snapshot.generation, "phase": control["phase"],
                    "pending": total - done - len(dispatch["active"]),
                    "active": len(dispatch["active"]), **{k: dispatch[k] for k in TERMINAL},
                    "elapsed_seconds": elapsed, "tasks_per_second": done / elapsed}

    @staticmethod
    def log(event, snapshot, **fields):
        print(json.dumps({"event": event, "group": snapshot.group_id,
                          "stage": snapshot.stage, "generation": snapshot.generation,
                          **fields}), flush=True)
