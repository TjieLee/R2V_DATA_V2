from types import SimpleNamespace

import pytest

from r2v_data_v2.h3.t2va_mimo_stage_server import (
    StageMimoClient,
    build_mimo_serve_command,
)


def client(tmp_path):
    return StageMimoClient(
        api_key="local-no-key",
        base_url="http://127.0.0.1:8092/v1",
        timeout_seconds=10,
        serve_command=["fake-sglang", "serve"],
        log_root=tmp_path,
        startup_polls=2,
        poll_interval=0,
        cleanup_grace_seconds=1,
    )


def test_mimo_serve_command_uses_full_stage_memory():
    command = build_mimo_serve_command(
        "/runtime/sglang", "/models/mimo", mem_fraction_static=0.65
    )
    assert command[:4] == [
        "/runtime/sglang",
        "serve",
        "--model-path",
        "/models/mimo",
    ]
    assert command[command.index("--mem-fraction-static") + 1] == "0.65"
    assert command[command.index("--tp") + 1] == "8"
    assert command[command.index("--dp") + 1] == "2"


def test_stage_without_requests_never_starts_mimo(tmp_path, monkeypatch):
    value = client(tmp_path)
    starts = []
    monkeypatch.setattr(
        value,
        "_ensure_started_locked",
        lambda: starts.append("start"),
    )
    with value.stage(17):
        assert value._process is None
    assert starts == []


def test_first_request_is_lazy_and_stage_cleanup_is_mandatory(tmp_path, monkeypatch):
    value = client(tmp_path)
    events = []

    def ensure():
        events.append("start")
        value._client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: kwargs)
            )
        )

    def stop():
        events.append("stop")
        value._client = None

    monkeypatch.setattr(value, "_ensure_started_locked", ensure)
    monkeypatch.setattr(value, "_stop_locked", stop)

    with value.stage(19):
        assert events == []
        assert value.chat.completions.create(marker="one") == {"marker": "one"}
        assert value.chat.completions.create(marker="two") == {"marker": "two"}
        assert events == ["start", "start"]
    assert events == ["start", "start", "stop"]


def test_busy_port_is_infrastructure_failure(tmp_path, monkeypatch):
    value = client(tmp_path)
    monkeypatch.setattr(value, "_port_available", lambda: False)
    with value.stage(23), pytest.raises(ConnectionError, match="connection refused"):
        value.chat.completions.create()
