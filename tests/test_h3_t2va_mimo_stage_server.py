from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event
from types import SimpleNamespace

import pytest

from r2v_data_v2.h3.t2va_mimo_stage_server import (
    StageMimoClient,
    StageMimoPoolClient,
    build_mimo_serve_command,
    mimo_v26_server_env,
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


def test_mimo_serve_command_keeps_v25_production_default():
    command = build_mimo_serve_command(
        "/runtime/sglang", "/models/mimo", mem_fraction_static=0.65
    )
    assert command[:4] == [
        "/runtime/sglang",
        "serve",
        "--model-path",
        "/models/mimo",
    ]
    assert command[command.index("--served-model-name") + 1] == "mimo-v2.5"
    assert command[command.index("--mem-fraction-static") + 1] == "0.65"
    assert command[command.index("--tp") + 1] == "8"
    assert command[command.index("--dp") + 1] == "2"
    assert "--speculative-algorithm" not in command


def test_mimo_v26_uses_validated_tp4_marlin_without_eagle():
    command = build_mimo_serve_command(
        "/runtime/sglang",
        "/models/mimo-v26",
        served_model_name="mimo-v2.6-flash-rl",
        port=8094,
    )
    assert command[command.index("--served-model-name") + 1] == "mimo-v2.6-flash-rl"
    assert command[command.index("--port") + 1] == "8094"
    assert command[command.index("--tp") + 1] == "4"
    assert command[command.index("--moe-runner-backend") + 1] == "marlin"
    assert "--disable-custom-all-reduce" in command
    assert not {"--dp", "--ep", "--speculative-algorithm", "--enable-multi-layer-eagle"} & set(command)


def test_v26_child_environment_is_local_to_its_gpu_group():
    child = mimo_v26_server_env("4,5,6,7")
    assert child["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert child["SGLANG_ENABLE_JIT_DEEPGEMM"] == "0"
    assert child["SGLANG_DEEPGEMM_PDL"] == "0"
    assert child["NO_PROXY"] == child["no_proxy"] == "127.0.0.1,localhost"


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
        if value._client is None:
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
        assert events == ["start"]
    assert events == ["start", "stop"]


def test_busy_port_is_infrastructure_failure(tmp_path, monkeypatch):
    value = client(tmp_path)
    monkeypatch.setattr(value, "_port_available", lambda: False)
    with value.stage(23), pytest.raises(ConnectionError, match="connection refused"):
        value.chat.completions.create()


def test_pool_leases_first_available_endpoint_and_returns_it_after_error():
    entered = [Event(), Event()]
    release = [Event(), Event()]

    class FakeEndpoint:
        def __init__(self, index):
            self.index = index
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create)
            )

        @contextmanager
        def stage(self, _shard_id):
            yield self

        def start(self):
            pass

        def create(self, **kwargs):
            entered[self.index].set()
            if not release[self.index].wait(2):
                raise TimeoutError("endpoint was not released")
            if kwargs.get("fail"):
                raise ValueError("clip failed")
            return self.index

    pool = StageMimoPoolClient([FakeEndpoint(0), FakeEndpoint(1)])
    with pool.stage(3), ThreadPoolExecutor(max_workers=3) as workers:
        first = workers.submit(pool.chat.completions.create)
        second = workers.submit(pool.chat.completions.create)
        assert entered[0].wait(2) and entered[1].wait(2)
        third = workers.submit(pool.chat.completions.create, fail=True)
        assert not third.done()
        release[1].set()
        assert second.result(timeout=2) == 1
        with pytest.raises(ValueError, match="clip failed"):
            third.result(timeout=2)
        assert pool.chat.completions.create(marker="next") == 1
        release[0].set()
        assert first.result(timeout=2) == 0
