from __future__ import annotations

import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE = "r2v_data_v2.h3.t2va_full_prefetch"
FACTORY = "tests.test_h3_t2va_full_prefetch:fake_prepare"


def until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("subprocess handshake timed out")


def alive(pid):
    result = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    return (
        result.returncode == 0
        and bool(result.stdout.strip())
        and not result.stdout.strip().startswith("Z")
    )


def fake_prepare(request):
    root = Path(request["root"])
    shard_id = request["shard_id"]
    mode = request["index"].get("mode", "ready")
    info = {
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "mask": list(signal.pthread_sigmask(signal.SIG_BLOCK, [])),
    }
    child = None
    if mode == "nested":
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)",
            ],
            start_new_session=True,
        )
        info["child"] = child.pid
    (root / f"entered-{shard_id}.json").write_text(json.dumps(info))
    try:
        if mode in {"blocked", "nested"}:
            until(lambda: (root / f"release-{shard_id}").exists(), timeout=60)
        if mode == "error":
            raise ValueError("synthetic canonical failure")
        if mode == "crash":
            os._exit(23)
        if mode == "invalid":
            return {"audio_root": "", "summary": {"ready": -1}}
        # Printing here must not corrupt the result-file protocol.
        print("fake ffmpeg output", flush=True)
        return {
            "audio_root": str(Path(request["shard_root"]) / "audio_production"),
            "summary": {"ready": 2, "failed": 1, "skipped": 3},
        }
    finally:
        if child is not None:
            try:
                child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


@pytest.fixture
def module():
    return importlib.import_module(MODULE)


def make(module, root, mode="ready", **kwargs):
    return module.CanonicalPrefetch(
        root,
        {"mode": mode},
        root / "clips",
        root / "videos",
        3,
        "fake-ffmpeg",
        "fake-ffprobe",
        worker_factory=FACTORY,
        **kwargs,
    )


def test_api_available():
    assert importlib.util.find_spec(MODULE) is not None


def test_owned_group_one_outstanding_and_order(module, tmp_path, capsys):
    before = dict(os.environ)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with make(module, tmp_path, "blocked") as prefetch:
        prefetch.start(4)
        entered = until(lambda: (tmp_path / "entered-4.json").exists())
        assert entered
        info = json.loads((tmp_path / "entered-4.json").read_text())
        assert info["pid"] == info["pgid"] != os.getpgrp()
        assert signal.SIGINT not in info["mask"] and signal.SIGTERM not in info["mask"]
        with pytest.raises(RuntimeError, match="outstanding"):
            prefetch.start(5)
        with pytest.raises(ValueError, match="shard"):
            prefetch.wait(5)
        assert not (tmp_path / "entered-5.json").exists()
        (tmp_path / "release-4").touch()
        result = prefetch.wait(4)
        assert result["summary"] == {"ready": 2, "failed": 1, "skipped": 3}
        assert not alive(info["pid"])
        (tmp_path / "release-5").touch()
        prefetch.start(5)
        assert prefetch.wait(5)["audio_root"] != result["audio_root"]
    assert dict(os.environ) == before
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())
    output = capsys.readouterr().out
    assert "shard=4 canonical_prefetch_started" in output
    assert "shard=4 canonical_prefetch_completed elapsed_seconds=" in output
    assert "fake ffmpeg output" not in output
    assert any(
        "fake ffmpeg output" in path.read_text() for path in tmp_path.rglob("*.log")
    )


def test_blocks_on_canonical_writer(module, tmp_path):
    from r2v_data_v2.h3.t2va_production import file_lock, shard_name

    with make(module, tmp_path) as prefetch:
        with file_lock(
            tmp_path / "shards" / shard_name(0) / "canonical.lock", blocking=True
        ):
            prefetch.start(0)
            log = tmp_path / "shards" / shard_name(0) / "logs/canonical-prefetch.log"
            until(lambda: "canonical_prefetch_started" in log.read_text())
            time.sleep(0.2)
            assert not (tmp_path / "entered-0.json").exists()
        assert prefetch.wait(0)["summary"]["ready"] == 2


def test_does_not_block_on_other_invocation_owner(module, tmp_path):
    from r2v_data_v2.h3.t2va_production import (
        ShardLockedError,
        file_lock,
        shard_name,
    )

    with make(module, tmp_path) as prefetch:
        with file_lock(
            tmp_path / "shards" / shard_name(0) / "invocation.lock", blocking=True
        ):
            prefetch.start(0)
            with pytest.raises(ShardLockedError, match="owned by another invocation"):
                prefetch.wait(0)
        assert not (tmp_path / "entered-0.json").exists()


@pytest.mark.parametrize(
    "mode,match",
    [("error", "synthetic canonical failure"), ("crash", "23"), ("invalid", "invalid")],
)
def test_worker_errors_are_not_success(module, tmp_path, mode, match):
    with make(module, tmp_path, mode) as prefetch:
        prefetch.start(0)
        with pytest.raises(RuntimeError, match=match):
            prefetch.wait(0)
        info = json.loads((tmp_path / "entered-0.json").read_text())
        assert not alive(info["pid"])


def test_close_cleans_detached_ffmpeg_like_child(module, tmp_path):
    prefetch = make(module, tmp_path, "nested")
    with prefetch:
        prefetch.start(0)
        until(lambda: (tmp_path / "entered-0.json").exists())
        info = json.loads((tmp_path / "entered-0.json").read_text())
        assert alive(info["child"])
        prefetch.close()
        until(lambda: not alive(info["child"]))
        assert not alive(info["pid"])
    prefetch.close()
    with pytest.raises(RuntimeError, match="closed"):
        prefetch.start(1)


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_supervisor_signal_cleans_owned_tree(module, tmp_path, sig):
    script = (
        f"from {MODULE} import CanonicalPrefetch; "
        f"from tests.test_h3_t2va_full_prefetch import FACTORY; "
        "import pathlib,time; "
        f"r=pathlib.Path({str(tmp_path)!r}); "
        "p=CanonicalPrefetch(r, {'mode':'nested'}, r/'clips', r/'videos', 2, 'ffmpeg', 'ffprobe', worker_factory=FACTORY); "
        "p.__enter__(); p.start(0); time.sleep(120)"
    )
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).resolve().parents[1])
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    }
    supervisor = subprocess.Popen(
        [sys.executable, "-c", script],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    info = None
    try:
        until(lambda: (tmp_path / "entered-0.json").exists())
        info = json.loads((tmp_path / "entered-0.json").read_text())
        supervisor.send_signal(sig)
        assert supervisor.wait(timeout=15) != 0
        until(lambda: not alive(info["child"]))
        assert not alive(info["pid"])
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.wait()
        if info:
            for pid in (info["child"], info["pid"]):
                if alive(pid):
                    os.kill(pid, signal.SIGKILL)


def test_spawn_is_registered_while_signals_masked(module, tmp_path, monkeypatch):
    popen = module.subprocess.Popen
    observed = []

    def spawn(*args, **kwargs):
        observed.append(signal.pthread_sigmask(signal.SIG_BLOCK, []))
        return popen(*args, **kwargs)

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    with make(module, tmp_path) as prefetch:
        prefetch.start(0)
        prefetch.wait(0)
    assert {signal.SIGINT, signal.SIGTERM} <= observed[0]


def test_default_worker_calls_production_api_only(module, tmp_path, monkeypatch):
    # Substitute only the import boundary, not any frozen producer implementation.
    calls = []
    audio = tmp_path / "shard/audio_production"
    selection = SimpleNamespace(shots=[1, 2, 3], excluded_rows=[{}])

    def bootstrap(shard, selected, backend, *, canonical_workers):
        calls.append((shard, selected, backend, canonical_workers))
        return audio

    full = SimpleNamespace(
        shard_selection=lambda *args: selection,
        bootstrap_audio=bootstrap,
        production=SimpleNamespace(complete_rows=lambda path: iter([{}, {}])),
    )
    media = SimpleNamespace(FFmpegAudioMediaBackend=lambda **kwargs: kwargs)
    monkeypatch.setitem(sys.modules, "r2v_data_v2.h3.t2va_full_production", full)
    monkeypatch.setitem(sys.modules, "r2v_data_v2.h3.audio_backends", media)
    result = module.prepare_canonical(
        {
            "root": str(tmp_path),
            "index": {},
            "shard_id": 0,
            "shard_root": str(tmp_path / "shard"),
            "clips_root": str(tmp_path / "clips"),
            "source_videos_root": None,
            "canonical_workers": 7,
            "ffmpeg": "ffm",
            "ffprobe": "ffp",
        }
    )
    assert calls == [
        (tmp_path / "shard", selection, {"ffmpeg": "ffm", "ffprobe": "ffp"}, 7)
    ]
    assert result == {
        "audio_root": str(audio),
        "summary": {"ready": 2, "failed": 1, "skipped": 1},
    }


@pytest.mark.parametrize("workers", [0, -1, True])
def test_invalid_worker_count(module, tmp_path, workers):
    with pytest.raises(ValueError, match="canonical_workers"):
        module.CanonicalPrefetch(
            tmp_path, {}, tmp_path, tmp_path, workers, "ffmpeg", "ffprobe"
        )


def test_completed_unconsumed_shard_still_owns_slot(module, tmp_path):
    with make(module, tmp_path) as prefetch:
        prefetch.start(0)
        until(lambda: list(tmp_path.rglob("result.json")))
        with pytest.raises(RuntimeError, match="outstanding"):
            prefetch.start(1)
        assert prefetch.wait(0)["summary"]["ready"] == 2


def test_signal_in_popen_registration_window(module, tmp_path, monkeypatch):
    popen = module.subprocess.Popen
    processes = []

    def spawn(*args, **kwargs):
        process = popen(*args, **kwargs)
        if kwargs.get("start_new_session"):
            processes.append(process)
            os.kill(os.getpid(), signal.SIGTERM)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    with pytest.raises(SystemExit) as error, make(module, tmp_path) as prefetch:
        prefetch.start(0)
    assert error.value.code == 128 + signal.SIGTERM
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before


def test_spawn_error_restores_mask_and_handlers(module, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("synthetic spawn failure")

    monkeypatch.setattr(module.subprocess, "Popen", fail)
    before = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    with (
        pytest.raises(OSError, match="spawn failure"),
        make(module, tmp_path) as prefetch,
    ):
        prefetch.start(0)
    assert signal.pthread_sigmask(signal.SIG_BLOCK, []) == before
    assert all(signal.getsignal(sig) == handler for sig, handler in handlers.items())


def test_parent_import_does_not_load_full_production():
    script = (
        "import sys; import r2v_data_v2.h3; before=set(sys.modules); "
        f"import {MODULE}; "
        "assert 'r2v_data_v2.h3.t2va_full_production' not in sys.modules; "
        "assert 'r2v_data_v2.h3.audio_backends' not in (set(sys.modules)-before); "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_context_exception_cleans_prefetch(module, tmp_path):
    info = None
    with (
        pytest.raises(ValueError, match="main stage failed"),
        make(module, tmp_path, "blocked") as prefetch,
    ):
        prefetch.start(0)
        until(lambda: (tmp_path / "entered-0.json").exists())
        info = json.loads((tmp_path / "entered-0.json").read_text())
        raise ValueError("main stage failed")
    assert info is not None and not alive(info["pid"])
