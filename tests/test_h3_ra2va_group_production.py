from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from tests.test_h3_ra2va_group_source import publish_group


def coordinator(root, source, max_groups=1):
    from r2v_data_v2.h3.ra2va_group_production import GroupCoordinator
    return GroupCoordinator(Path(root), Path(source), 20, max_groups=max_groups)


def make_run(tmp_path, count=20, max_groups=1):
    source = tmp_path / "post_mask"
    publish_group(source, count=count)
    c = coordinator(tmp_path / "run", source, max_groups)
    return c, c.select_current()


def drain(root, source, executions):
    c = coordinator(root, source)
    s = c.select_current()
    while claim := c.claim(s, str(mp.current_process().pid)):
        with claim:
            executions.put(claim.task.ordinal)
            c.publish(claim, "ready", {"fake": True}, 0.001)


def crash_worker(root, source, point, reached, release):
    from r2v_data_v2.h3 import ra2va_group_production as module
    c = coordinator(root, source)
    s = c.select_current()
    original = module.atomic_json

    def write(path, value):
        is_dispatch = path.name == "dispatch.json"
        if point == "before_dispatch" and is_dispatch and value["active"]:
            reached.set()
            release.wait(30)
        original(path, value)
        if point == "after_result" and path.parent.name == "results":
            reached.set()
            release.wait(30)

    module.atomic_json = write
    claim = c.claim(s, "crashing")
    with claim:
        if point == "after_claim":
            reached.set()
            release.wait(30)
        c.publish(claim, "ready", {"from_crash": True}, 0.01)


def test_multiprocess_claim_and_publish_once(tmp_path):
    c, s = make_run(tmp_path)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    workers = [ctx.Process(target=drain, args=(c.run_root, c.source_root, queue)) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(15)
        assert not worker.is_alive()
        assert worker.exitcode == 0
    executed = [queue.get(timeout=2) for _ in range(20)]
    assert sorted(executed) == list(range(20))
    stats = c.progress(s)
    assert stats["ready"] == 20 and stats["pending"] == stats["active"] == 0
    assert len(list(c.stage_root(s).glob("results/*.json"))) == 20


@pytest.mark.parametrize("point", ["before_dispatch", "after_claim", "after_result"])
def test_event_controlled_crash_recovery(tmp_path, point):
    c, s = make_run(tmp_path, count=1)
    ctx = mp.get_context("spawn")
    reached, release = ctx.Event(), ctx.Event()
    worker = ctx.Process(target=crash_worker, args=(c.run_root, c.source_root, point, reached, release))
    worker.start()
    try:
        assert reached.wait(10)
        worker.kill()
        worker.join(5)
        assert not worker.is_alive()
        c.recover(s)
        claim = c.claim(s, "resumed")
        if point == "after_result":
            assert claim is None
            assert json.loads(c.result_path(s, 0).read_text())["payload"]["from_crash"]
        else:
            assert claim.task.ordinal == 0
            with claim:
                c.publish(claim, "ready", {"resumed": True}, 0.01)
        assert c.progress(s)["ready"] == 1
        assert c.progress(s)["active"] == c.progress(s)["pending"] == 0
    finally:
        if worker.is_alive():
            worker.kill()
            worker.join()


def test_live_clip_lock_is_not_reclaimed(tmp_path):
    c, s = make_run(tmp_path, count=2)
    held = c.claim(s, "live")
    try:
        c.recover(s)
        other = c.claim(s, "other")
        assert other.task.ordinal == 1
        with other:
            c.publish(other, "ready", {}, 0)
        assert c.claim(s, "third") is None
    finally:
        held.close()
    recovered = c.claim(s, "resumed")
    assert recovered.task.ordinal == 0
    with recovered:
        c.publish(recovered, "ready", {}, 0)


def test_terminal_failed_and_skipped_are_not_retried(tmp_path):
    c, s = make_run(tmp_path, count=2)
    for status in ("failed", "skipped"):
        with c.claim(s, "one") as claim:
            c.publish(claim, status, {}, 0)
    resumed = coordinator(c.run_root, c.source_root)
    resumed.recover(s)
    assert resumed.claim(s, "two") is None
    assert resumed.progress(s)["failed"] == resumed.progress(s)["skipped"] == 1


def test_conflicting_resume_settings_are_rejected(tmp_path):
    c, _ = make_run(tmp_path)
    with pytest.raises(ValueError, match="settings"):
        coordinator(c.run_root, c.source_root, max_groups=2)


def complete_stage(c, s):
    while claim := c.claim(s, "drain"):
        with claim:
            c.publish(claim, "ready", {}, 0)


def complete_group(c):
    s = c.select_current()
    group = s.group_id
    while s and s.group_id == group:
        complete_stage(c, s)
        assert c.advance(s)
        s = c.select_current()
    return s


def test_barrier_waits_for_live_worker_release(tmp_path):
    c, s = make_run(tmp_path, count=1)
    session = c.join(s, "live")
    try:
        complete_stage(c, s)
        assert not c.advance(s)
        assert c.select_current().phase == "draining"
        c.acknowledge_release(session)
        assert c.advance(s)
        assert c.select_current().stage == "sam"
    finally:
        session.close()


def test_dead_session_does_not_block_barrier(tmp_path):
    c, s = make_run(tmp_path, count=1)
    session = c.join(s, "departed")
    session.close()
    complete_stage(c, s)
    assert c.advance(s)


def test_late_join_is_rejected_while_draining(tmp_path):
    c, s = make_run(tmp_path, count=1)
    session = c.join(s, "live")
    try:
        complete_stage(c, s)
        assert not c.advance(s)
        with pytest.raises(RuntimeError, match="draining"):
            c.join(s, "late")
    finally:
        session.close()


def test_stale_generation_cannot_claim_publish_or_ack(tmp_path):
    c, s = make_run(tmp_path, count=1)
    session = c.join(s, "old")
    claim = c.claim(s, "old")
    try:
        c.publish(claim, "ready", {}, 0)
        c.acknowledge_release(session)
        assert c.advance(s)
        before = c.control_path.read_bytes()
        for mutation in (lambda: c.claim(s, "old"), lambda: c.publish(claim, "ready", {}, 0),
                         lambda: c.recover(s), lambda: c.acknowledge_release(session)):
            with pytest.raises(RuntimeError, match="generation"):
                mutation()
        assert c.control_path.read_bytes() == before
    finally:
        claim.close()
        session.close()


def test_started_group_wins_over_lower_new_group(tmp_path):
    from r2v_data_v2.h3.ra2va_group_source import (
        discover_published_groups,
        freeze_pilot_inventory,
    )
    source = tmp_path / "post_mask"
    publish_group(source, "group-000018", count=1)
    c = coordinator(tmp_path / "run", source)
    g = discover_published_groups(source)[0]
    freeze_pilot_inventory(g, c.group_root(g.group_id) / "inventory", 20)
    publish_group(source, "group-000000", count=1)
    assert c.select_current().group_id == "group-000018"


def test_new_group_discovered_while_running_and_global_limit(tmp_path):
    c, _ = make_run(tmp_path, count=1, max_groups=2)
    publish_group(c.source_root, "group-000001", count=1)
    next_stage = complete_group(c)
    assert next_stage.group_id == "group-000001"
    assert complete_group(c) is None
    publish_group(c.source_root, "group-000002", count=1)
    restarted = coordinator(c.run_root, c.source_root, max_groups=2)
    assert restarted.select_current() is None
    assert read_control(c)["started_groups"] == ["group-000000", "group-000001"]
    assert not (c.group_root("group-000002") / "inventory").exists()
    assert not list(c.run_root.rglob("COMPLETE"))
    assert len(list(c.run_root.rglob("PILOT_COMPLETE"))) == 2


def read_control(c):
    return json.loads(c.control_path.read_text())


def test_stage_restart_keeps_generation(tmp_path):
    c, s = make_run(tmp_path, count=1)
    restarted = coordinator(c.run_root, c.source_root)
    assert restarted.select_current() == s


def test_crash_during_generation_advance(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import ra2va_group_production as module
    c, s = make_run(tmp_path, count=1)
    complete_stage(c, s)
    original = module.atomic_json

    def write(path, value):
        if path == c.control_path and value["stage"] == "sam":
            raise OSError("interrupted before control publication")
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_json", write)
        with pytest.raises(OSError):
            c.advance(s)
    assert c.select_current().generation == s.generation
    assert c.advance(s)
    assert c.select_current().generation == s.generation + 1
    assert c.progress(s.__class__(s.group_id, "sam", s.generation + 1, "running"))["pending"] == 1


def test_resume_after_pilot_completion_before_next_group(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import ra2va_group_production as module
    c, s = make_run(tmp_path, count=1)
    while s.stage != "export":
        complete_stage(c, s)
        c.advance(s)
        s = c.select_current()
    complete_stage(c, s)
    original = module.atomic_json

    def write(path, value):
        if path == c.control_path and value["phase"] == "complete":
            raise OSError("interrupted after PILOT_COMPLETE")
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(module, "atomic_json", write)
        with pytest.raises(OSError):
            c.advance(s)
    restarted = coordinator(c.run_root, c.source_root)
    assert restarted.select_current() is None
    assert restarted.progress(s)["ready"] == 1
