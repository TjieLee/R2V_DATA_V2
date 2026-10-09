from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.test_h3_mimo25_av_shadow import _Completions
from tests.test_h3_mimo26_two_step import _drafts, _reconcile, _setup


def client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def run(root, config, completions, job, stems, **kwargs):
    from r2v_data_v2.h3.ra2va_group_mimo import reconcile_group_job
    return reconcile_group_job(root=root, config=config, client=client(completions), job=job,
        stems={s.stem_type: s.canonical_stem_path for s in stems[0].stems}, **kwargs)


def test_durable_turns_preserve_exact_requests_raw_and_historical_usage(tmp_path, monkeypatch):
    _, _, backend, original, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    expected = _reconcile(backend, jobs[0], stems)
    visual, joint = _drafts()
    completions = _Completions([(json.dumps(visual), 0), (json.dumps(joint), 12)])
    root = tmp_path / "durable"
    result = run(root, backend.config, completions, jobs[0], stems)
    assert result["status"] == "ready", result
    assert completions.requests == original.requests
    assert result["raw_responses"] == list(expected.raw_responses)
    assert result["model_call_count"] == result["new_request_count"] == 2
    assert result["response_replay_count"] == 0
    saved = [(root / f"{turn}.response.json").read_bytes() for turn in ("visual", "joint")]
    replay = _Completions([])
    resumed = run(root, backend.config, replay, jobs[0], stems)
    assert replay.requests == []
    assert resumed["annotation"] == result["annotation"]
    assert resumed["diagnostics"] == result["diagnostics"]
    assert resumed["model_call_count"] == 2
    assert resumed["new_request_count"] == 0 and resumed["response_replay_count"] == 2
    assert saved == [(root / f"{turn}.response.json").read_bytes() for turn in ("visual", "joint")]


def test_saved_visual_resume_only_requests_joint(tmp_path, monkeypatch):
    _, _, backend, _, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    visual, joint = _drafts()
    completions = _Completions([(json.dumps(visual), 0)])

    def crash(event, turn):
        if event == "response_saved" and turn == "visual":
            raise SystemExit("interrupted after durable Visual")

    root = tmp_path / "resume"
    with pytest.raises(SystemExit, match="durable Visual"):
        run(root, backend.config, completions, jobs[0], stems, event=crash)
    assert len(completions.requests) == 1
    resumed_calls = _Completions([(json.dumps(joint), 12)])
    result = run(root, backend.config, resumed_calls, jobs[0], stems)
    assert result["status"] == "ready", result
    assert len(resumed_calls.requests) == result["new_request_count"] == 1
    assert result["model_call_count"] == 2 and result["response_replay_count"] == 1


def test_unresolved_request_never_reissues_model_call(tmp_path, monkeypatch):
    _, _, backend, _, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)

    class Interrupted:
        def create(self, **payload):
            raise SystemExit("process killed in HTTP request")

    root = tmp_path / "unknown"
    with pytest.raises(SystemExit):
        run(root, backend.config, Interrupted(), jobs[0], stems)
    completions = _Completions([])
    result = run(root, backend.config, completions, jobs[0], stems)
    assert result["status"] == "failed"
    assert result["failure_code"] == "interrupted_request_unresolved"
    assert result["unresolved_turns"] == ["visual"]
    assert result["new_request_count"] == 0 and completions.requests == []


def test_invalid_structured_raw_is_saved_and_not_requested_again(tmp_path, monkeypatch):
    _, _, backend, _, stems, jobs, _, _ = _setup(tmp_path, monkeypatch)
    completions = _Completions([('{"bad": true}', 0)])
    root = tmp_path / "invalid"
    first = run(root, backend.config, completions, jobs[0], stems)
    assert first["status"] == "failed"
    assert first["visual_raw_response"] == '{"bad": true}'
    assert (root / "visual.response.json").is_file()
    replay = _Completions([])
    second = run(root, backend.config, replay, jobs[0], stems)
    assert second["failure_code"] == first["failure_code"]
    assert second["visual_raw_response"] == first["visual_raw_response"]
    assert replay.requests == [] and second["model_call_count"] == 1
