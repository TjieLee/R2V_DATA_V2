from __future__ import annotations

import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_h3_ra2va_group_production import make_run
from tests.test_h3_ra2va_group_source import publish_group


def run(source, root, workers=4, max_groups=1):
    from r2v_data_v2.h3.ra2va_group_fake import run_fake_pilot
    return run_fake_pilot(source_root=Path(source), run_root=Path(root), workers=workers,
                          limit=20, max_groups=max_groups, poll_seconds=0.01)


def test_eight_fake_stages_and_resume_without_reexecution(tmp_path):
    source = tmp_path / "post_mask"
    publish_group(source)
    root = tmp_path / "run"
    result = run(source, root)
    assert result["completed_groups"] == 1 and result["model_call_count"] == 0
    results = list(root.glob("groups/*/stages/*/results/*.json"))
    assert len(results) == 160
    assert all(json.loads(p.read_text())["payload"]["fake"] for p in results)
    before = {p: p.read_bytes() for p in results}
    publish_group(source, "group-000001")
    assert run(source, root)["completed_groups"] == 1
    assert before == {p: p.read_bytes() for p in results}
    assert not (root / "groups/group-000001").exists()
    assert not list(root.rglob("COMPLETE"))


def test_terminal_failure_propagates_without_retry(tmp_path):
    c, s = make_run(tmp_path, count=2)
    with c.claim(s, "business_failure") as claim:
        c.publish(claim, "failed", {"reason": "fake business failure"}, 0)
    run(c.source_root, c.run_root)
    results = [json.loads(p.read_text()) for p in c.run_root.glob("groups/*/stages/*/results/000000.json")]
    assert len(results) == 8
    assert sum(r["status"] == "failed" for r in results) == 1
    assert sum(r["status"] == "skipped" for r in results) == 7


def test_two_launchers_obey_persisted_global_group_limit(tmp_path):
    source = tmp_path / "post_mask"
    for i in range(3):
        publish_group(source, f"group-{i:06d}", count=3)
    root = tmp_path / "run"
    ctx = mp.get_context("spawn")
    launchers = [ctx.Process(target=run, args=(source, root, 2, 1)) for _ in range(2)]
    try:
        for p in launchers:
            p.start()
        for p in launchers:
            p.join(20)
            assert not p.is_alive() and p.exitcode == 0
    finally:
        for p in launchers:
            if p.is_alive():
                p.kill()
                p.join()
    assert len(list(root.glob("groups/*/PILOT_COMPLETE"))) == 1
    assert len(list(root.glob("groups/*/inventory"))) == 1
    assert len(list(root.glob("groups/*/stages/*/results/*.json"))) == 24


def test_fake_context_releases_before_ack(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import ra2va_group_fake as module
    c, _ = make_run(tmp_path, count=1)
    live = set()
    original_enter, original_exit = module.FakeWorker.__enter__, module.FakeWorker.__exit__
    original_ack = c.acknowledge_release

    def enter(worker):
        live.add(worker.stage)
        return original_enter(worker)

    def exit(worker, *args):
        original_exit(worker, *args)
        live.remove(worker.stage)

    def acknowledge(session):
        assert not live
        original_ack(session)

    monkeypatch.setattr(module.FakeWorker, "__enter__", enter)
    monkeypatch.setattr(module.FakeWorker, "__exit__", exit)
    monkeypatch.setattr(c, "acknowledge_release", acknowledge)
    module.run_fake_worker(c, "local", mp.get_context("spawn").Event(), poll_seconds=0.001)
    assert not live


def test_cli_fake_only_and_output_boundary(tmp_path):
    from r2v_data_v2.h3.ra2va_group_fake import pilot_run_root
    authorized = Path("/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA")
    assert pilot_run_root(authorized, "pilot-20") == authorized / "pilot-20"
    with pytest.raises(ValueError):
        pilot_run_root(authorized.parent / "in_pair_reference", "pilot-20")
    with pytest.raises(ValueError):
        pilot_run_root(tmp_path, "../elsewhere")
    cli = Path(__file__).parents[1] / "tools/run_h3_ra2va_group_production.py"
    result = subprocess.run([sys.executable, str(cli), "run", "--limit", "20"],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode != 0 and "--fake" in result.stderr


def test_no_model_imports():
    from r2v_data_v2.h3.ra2va_group_fake import FakeWorker
    with FakeWorker("mimo"):
        assert "torch" not in sys.modules
        assert "r2v_data_v2.h3.mimo26_two_step_backend" not in sys.modules


def test_join_race_with_draining_is_not_infrastructure_failure(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import ra2va_group_fake as module
    from tests.test_h3_ra2va_group_production import complete_stage
    c, s = make_run(tmp_path, count=1)
    blocker = c.join(s, "blocker")
    original = c.join
    first = True

    def join(snapshot, worker_id):
        nonlocal first
        if first:
            first = False
            complete_stage(c, s)
            assert not c.advance(s)
            blocker.close()
        return original(snapshot, worker_id)

    monkeypatch.setattr(c, "join", join)
    module.run_fake_worker(c, "racing", mp.get_context("spawn").Event(), poll_seconds=0.001)
    assert (c.group_root(s.group_id) / "PILOT_COMPLETE").exists()


def test_completed_stage_elapsed_is_stable(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import ra2va_group_fake as module
    source = tmp_path / "post_mask"
    publish_group(source, count=1)
    root = tmp_path / "run"
    before = run(source, root, workers=1)
    now = module.time.time()
    monkeypatch.setattr(module.time, "time", lambda: now + 100)
    after = module.pilot_summary(root)
    assert [s["elapsed_seconds"] for s in before["stages"]] == [s["elapsed_seconds"] for s in after["stages"]]
