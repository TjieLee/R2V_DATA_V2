from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.h3_ra2va_group_http_helpers import (
    Clock,
    claim,
    connect,
    post,
    release,
    result_body,
)
from tests.test_h3_ra2va_group_http_native import native_service
from tests.test_h3_ra2va_group_source import publish_group


def mimo_service(tmp_path, count=2):
    source = tmp_path / "post_mask"
    publish_group(source, count=count)
    service = native_service(tmp_path / "run", source, limit=count, wall_clock=Clock())
    for _ in range(6):
        session = connect(service, execution_mode="cpu_fixture")
        while (task := claim(service, session)).get("task") is not None:
            post(service, "/v1/tasks/result", **result_body(task))
        service.tick()
        release(service, session)
        service.tick()
    assert service.snapshot.stage == "mimo"
    return service


def turn_event(service, task, turn, event):
    root = Path(task["artifact_root"]) / "mimo" / task["task"]["clip_uid"]
    return post(service, "/v1/tasks/event", **{
        k: task[k] for k in ("session_id", "task_id", "claim_token")},
        event=event, turn=turn, checkpoint={"root": str(root), "turn": turn})


def save_response(task, turn):
    root = Path(task["artifact_root"]) / "mimo" / task["task"]["clip_uid"]
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{turn}.response.json"
    path.write_text(json.dumps({"clip_uid": task["task"]["clip_uid"], "turn": turn,
        "completion": {"choices": [{"message": {"content": "unchanged raw"}}]}}))
    return root


def test_saved_visual_reclaimed_after_response_event_reply_was_lost(tmp_path):
    service = mimo_service(tmp_path)
    try:
        old = claim(service, connect(service, "old", execution_mode="cpu_fixture"))
        turn_event(service, old, "visual", "request_started")
        root = save_response(old, "visual")
        original = (root / "visual.response.json").read_bytes()
        service.wall_clock.now += 61
        service.tick()
        new = claim(service, connect(service, "new", execution_mode="cpu_fixture"))
        assert new["task_id"] == old["task_id"]
        assert new["artifact_root"] != old["artifact_root"]
        assert new["replay_roots"] == {"visual": str(root)}
        other = claim(service, connect(service, "other", execution_mode="cpu_fixture"))
        assert other["task_id"]["ordinal"] == 1 and not other.get("replay_roots")
        assert service.dispatch("POST", "/v1/tasks/result", result_body(old))[0] == 409
        assert (root / "visual.response.json").read_bytes() == original
    finally:
        service.close()


def test_native_intent_without_response_is_terminal_unresolved(tmp_path):
    service = mimo_service(tmp_path, count=1)
    try:
        old = claim(service, connect(service, execution_mode="cpu_fixture"))
        snapshot = service.snapshot
        turn_event(service, old, "visual", "request_started")
        service.wall_clock.now += 61
        service.tick()
        result = json.loads(service.core.result_path(snapshot, 0).read_text())
        assert result["status"] == "failed"
        assert result["payload"]["failure_code"] == "interrupted_request_unresolved"
        assert result["payload"]["unresolved_turns"] == ["visual"]
        assert "fake" not in result["payload"]
        assert result["payload"]["model_call_count"] == 1
        assert not service._dispatch_state()["active"]
        service.tick()
        assert service.snapshot.stage == "export"
    finally:
        service.close()


def test_native_event_checkpoint_cannot_point_at_another_clip_or_turn(tmp_path):
    service = mimo_service(tmp_path)
    try:
        a, b = [claim(service, connect(service, name, execution_mode="cpu_fixture")) for name in ("a", "b")]
        identity = {k: a[k] for k in ("session_id", "task_id", "claim_token")}
        for checkpoint in ({"root": b["artifact_root"], "turn": "visual"},
                           {"root": a["artifact_root"], "turn": "joint"}):
            assert service.dispatch("POST", "/v1/tasks/event", identity | {
                "event": "request_started", "turn": "visual", "checkpoint": checkpoint})[0] == 409
        assert service._dispatch_state()["active"]["0"]["http"]["events"] == {}
        turn_event(service, a, "visual", "request_started")
        turn_event(service, a, "visual", "request_started")
        assert len(service._dispatch_state()["active"]["0"]["http"]["events"]) == 1
    finally:
        service.close()


@pytest.mark.parametrize("saved_turns", [("visual",), ("visual", "joint")])
def test_durable_replay_reads_only_authorized_old_attempts(tmp_path, monkeypatch, saved_turns):
    from tests.test_h3_mimo25_av_shadow import _Completions
    from tests.test_h3_mimo26_two_step import _drafts, _setup
    from tests.test_h3_ra2va_group_mimo import run
    _, _, backend, _, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    visual, joint = _drafts()
    old, new = tmp_path / "old-claim", tmp_path / "new-claim"

    def crash(event, turn):
        if event == "response_saved" and turn == saved_turns[-1]:
            raise SystemExit("after saved response")
    completions = _Completions([(json.dumps(visual), 0), (json.dumps(joint), 12)])
    with pytest.raises(SystemExit):
        run(old, backend.config, completions, jobs[0], stems, event=crash)
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    remaining = [] if len(saved_turns) == 2 else [(json.dumps(joint), 12)]
    calls = _Completions(remaining)
    result = run(new, backend.config, calls, jobs[0], stems,
                 replay_roots={turn: str(old) for turn in saved_turns})
    assert result["status"] == "ready", result
    assert result["model_call_count"] == 2
    assert result["new_request_count"] == 2 - len(saved_turns) == len(calls.requests)
    assert result["response_replay_count"] == len(saved_turns)
    assert before == {p.name: p.read_bytes() for p in old.iterdir()}
    assert result["response_sources"]["visual"] == str(old / "visual.response.json")


def test_request_event_must_be_confirmed_before_model_request(tmp_path, monkeypatch):
    from tests.test_h3_mimo25_av_shadow import _Completions
    from tests.test_h3_mimo26_two_step import _drafts, _setup
    from tests.test_h3_ra2va_group_mimo import run
    _, _, backend, _, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    visual, _ = _drafts()
    calls = _Completions([(json.dumps(visual), 0)])

    def no_ack(event, turn):
        assert event == "request_started" and turn == "visual"
        raise SystemExit("no durable Coordinator confirmation")
    with pytest.raises(SystemExit):
        run(tmp_path / "claim", backend.config, calls, jobs[0], stems, event=no_ack)
    assert calls.requests == []


def test_coordinator_restart_preserves_unexpired_per_turn_claim(tmp_path):
    service = mimo_service(tmp_path, count=1)
    old = claim(service, connect(service, execution_mode="cpu_fixture"))
    turn_event(service, old, "visual", "request_started")
    save_response(old, "visual")
    turn_event(service, old, "visual", "response_saved")
    clock = service.wall_clock
    service.close()
    restarted = native_service(tmp_path / "run", tmp_path / "post_mask", limit=1, wall_clock=clock)
    try:
        assert connect(restarted, execution_mode="cpu_fixture")["claim"] == old
        assert claim(restarted, connect(restarted, "b", execution_mode="cpu_fixture")).get("task") is None
        clock.now += 61
        restarted.tick()
        new = claim(restarted, connect(restarted, "b", execution_mode="cpu_fixture"))
        assert new["replay_roots"] == {"visual": str(Path(old["artifact_root"]) / "mimo/clip-0")}
        assert restarted.dispatch("POST", "/v1/tasks/event", {
            k: old[k] for k in ("session_id", "task_id", "claim_token")} | {"event": "execution_started"})[0] == 409
    finally:
        restarted.close()
