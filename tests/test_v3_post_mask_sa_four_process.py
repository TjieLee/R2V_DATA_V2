"""Execution-only SA 4-process dispatch and restart-order regression tests."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from r2v_data_v2.v3 import post_mask_epoch_production as production
from r2v_data_v2.v3.post_mask_epoch_sa_execution import (
    SASamPipelineExecutor, StageAwareSASamExecutor,
)
from r2v_data_v2.v3.post_mask_epoch_sa_process import (
    SAProcessHandle, SAProcessInferenceError, SAProcessTransportFatal,
)
from r2v_data_v2.v3.post_mask_epoch_groups import build_groups
from tools.prepare_v3_post_mask_sa_priority import build_sa_group_priority


def _job(clip: str, serial: int):
    return SimpleNamespace(
        canonical_shard="shard-000000000-000009999",
        clip_uid=clip, job_type="sam", attempt_index=0,
        job_id=lambda: f"{clip}-{serial:03d}",
    )


class _Runner:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []
        self.lock = threading.Lock()

    def prepare_sam_job(self, job):
        return job

    def infer_sam_job(self, job, value, handle):
        with self.lock:
            self.calls.append((job.clip_uid, handle))
        return handle

    def persist_sam_job(self, job, value):
        return value


def _collect_n(executor, n):
    result = []
    while len(result) < n:
        result.extend(executor.collect())
    return result


def test_sa_multi_slot_affinity_keeps_one_clip_on_one_process():
    runner = _Runner()
    resource = SimpleNamespace(
        gpu_ids=(0,), affinity_by_clip=True,
        handle_for_slot=lambda i: i,
    )
    executor = SASamPipelineExecutor(
        runner, resource=resource, slot_count=4, prepare_workers=2,
        persist_workers=4,
    )
    try:
        jobs = [_job(f"clip-{i % 4}", i) for i in range(12)]
        for job in jobs:
            executor.submit(job)
        completed = _collect_n(executor, len(jobs))
        assert len(completed) == 12
        grouped = {}
        for clip, worker in runner.calls:
            grouped.setdefault(clip, set()).add(worker)
        assert len(grouped) == 4
        assert all(len(workers) == 1 for workers in grouped.values())
        assert executor.diagnostics()["model_processes"] == 4
        assert executor.diagnostics()["physical_slots"] == 1
    finally:
        executor.close()


def test_stage_aware_four_process_handoff_is_sa_only(monkeypatch):
    monkeypatch.setenv("POST_MASK_SA_SAM_PROCESSES_PER_GPU", "4")
    monkeypatch.setenv("POST_MASK_SA_SAM_PERSIST_WORKERS", "4")
    opened = []

    class FakePool:
        affinity_by_clip = True
        slot_count = 4
        gpu_ids = (0,)

        def __init__(self, *args, **kwargs):
            self.closed = False
            opened.append(self)

        def start(self):
            pass

        def close(self):
            self.closed = True

        def handle_for_slot(self, slot):
            return f"process-{slot}"

    from r2v_data_v2.v3 import post_mask_epoch_sa_process
    monkeypatch.setattr(post_mask_epoch_sa_process, "SAProcessPool", FakePool)
    releases = []
    resource = SimpleNamespace(
        handle_for_slot=lambda slot: "old-sam",
        release_workers_for_sa=lambda: releases.append(True),
    )
    stage = [None]
    runner = _Runner()
    executor = StageAwareSASamExecutor(
        lambda job, handle: handle,
        pipeline_runner=lambda: stage[0],
        slot_count=1, resource=resource,
        sam_config=object(), gpu_ids=(0,),
    )
    try:
        assert executor.capacity() == 1
        executor.submit(_job("reference", 0))
        assert executor.collect()[0].result == "old-sam"
        assert not releases
        stage[0] = runner
        assert executor.capacity() == 12
        assert len(releases) == 1
        jobs = [_job(f"clip-{i % 4}", i) for i in range(12)]
        for job in jobs:
            executor.submit(job)
        assert len(_collect_n(executor, len(jobs))) == 12
        assert all(handle.startswith("process-") for _, handle in runner.calls)
    finally:
        executor.close()
    assert opened and opened[0].closed


def test_process_handle_model_exception_vs_fatal_transport():
    class Conn:
        def __init__(self, response):
            self.response = response
            self.sent = []

        def send(self, value):
            self.sent.append(value)

        def poll(self, timeout):
            return True

        def recv(self):
            return self.response

    process = SimpleNamespace(pid=123, is_alive=lambda: True)
    success = SAProcessHandle(Conn(("ok", ("mask",))), process, timeout_seconds=3)
    assert success.segment_frame(
        frame_path=Path("/tmp/f.jpg"), frame_slot=2, grounding_prompt="coat"
    ) == ("mask",)

    ordinary = SAProcessHandle(
        Conn(("error", ("ValueError", "bad prompt"))), process, timeout_seconds=3
    )
    with pytest.raises(SAProcessInferenceError, match="ValueError: bad prompt"):
        ordinary.segment_generated_frame(
            frame_path=Path("/tmp/g.jpg"), grounding_prompt="coat"
        )
    fatal = SAProcessHandle(
        Conn(("fatal", ("RuntimeError", "CUDA process died"))),
        process, timeout_seconds=3,
    )
    with pytest.raises(SAProcessTransportFatal):
        fatal.segment_generated_frame(
            frame_path=Path("/tmp/g.jpg"), grounding_prompt="coat"
        )
    assert not issubclass(SAProcessTransportFatal, Exception)


def test_sa_progress_snapshot_and_global_claim_priority(tmp_path):
    state = tmp_path / "state" / "resource_epochs"
    state.mkdir(parents=True)
    groups = build_groups(
        [f"shard-{i:09d}-{i:09d}" for i in range(4)],
        campaign={"source": "synthetic"}, group_size=1,
    )
    # Incomplete started groups, with different terminal percentages.
    for i, terminals in ((0, 2), (1, 8)):
        group = state / groups[i].group_id
        composition = group / "composition"
        composition.mkdir(parents=True)
        shard = groups[i].canonical_shards[0]
        (composition / "subject_attributes_started.json").write_text(
            json.dumps({"eligible_clip_uids_by_shard": {shard: list(range(10))}}),
            encoding="utf-8",
        )
        outcomes = group / "semantic" / "subject_attributes" / "outcomes" / shard
        outcomes.mkdir(parents=True)
        for j in range(terminals):
            (outcomes / f"clip-{j}.json").write_text("{}")
    done = state / groups[2].group_id / "composition"
    done.mkdir(parents=True)
    (done / "subject_attributes_completed.json").write_text("{}")
    (state / groups[3].group_id).mkdir(parents=True)
    payload = build_sa_group_priority(state)
    assert payload["ordered_group_ids"] == [
        groups[2].group_id, groups[1].group_id,
        groups[0].group_id, groups[3].group_id,
    ]
    # Explicit ordering must survive rank rotation when nodes shrink.
    snapshot = tmp_path / "priority.json"
    snapshot.write_text(json.dumps({
        "ordered_group_ids": payload["ordered_group_ids"],
    }))
    visited = []

    def runner(group, _ledger, _emit):
        visited.append(group.group_id)
        return {
            "completed": True,
            "subject_attributes_completed": True,
            "export_completed": True,
            "export_stats": {
                shard: {"sample_count": 0} for shard in group.canonical_shards
            },
        }

    production.run_elastic_groups(
        tmp_path / "state", groups, runner,
        rank=2, world_size=3, sleep=lambda _: None,
        priority_file=snapshot,
    )
    assert visited == payload["ordered_group_ids"]


def test_sa_priority_file_cannot_change_group_membership(tmp_path):
    groups = build_groups(
        ["shard-a"], campaign={"source": "synthetic"}, group_size=1,
    )
    file = tmp_path / "wrong.json"
    file.write_text(json.dumps({"ordered_group_ids": ["not-a-real-group"]}))
    with pytest.raises(ValueError, match="unknown group IDs"):
        production.run_elastic_groups(
            tmp_path / "state", groups, lambda *_: {},
            rank=0, world_size=1, priority_file=file,
        )
