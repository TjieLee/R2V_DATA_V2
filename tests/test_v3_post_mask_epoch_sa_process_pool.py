import errno
import importlib
import multiprocessing
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r2v_data_v2.v3.post_mask_epoch_resources import (
    EpochResourceError,
    WorkerPoolConfig,
)

_FRAME_ROOT = Path()


@pytest.fixture(autouse=True)
def _frame_root(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_FRAME_ROOT", tmp_path)


class FakeSegmenter:
    def __init__(self, config):
        self.config = config
        self.calls = 0
        assert config.device == "cuda:0"
        assert "torch" not in sys.modules
        assert not any(key in os.environ for key in
                       ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"))

    def segment_frame(self, *, frame_path, frame_slot, grounding_prompt):
        self.calls += 1
        if grounding_prompt == "crash":
            os._exit(23)
        if grounding_prompt == "hang":
            time.sleep(30)
        if grounding_prompt == "slow":
            time.sleep(0.15)
        if grounding_prompt == "value":
            raise ValueError("bad clip")
        if grounding_prompt == "type":
            raise TypeError("bad clip")
        if grounding_prompt == "triton":
            raise TypeError("'NoneType' object is not a mapping")
        if grounding_prompt == "missing":
            raise FileNotFoundError("missing clip")
        if grounding_prompt == "permission":
            raise PermissionError("denied")
        if grounding_prompt == "io":
            raise OSError(errno.EIO, "storage failed")
        if grounding_prompt == "oom":
            raise RuntimeError("CUDA out of memory")
        if grounding_prompt == "interrupt":
            raise KeyboardInterrupt("stop")
        if grounding_prompt == "exit":
            raise SystemExit(7)
        return (np.array([os.getpid(), int(os.environ["CUDA_VISIBLE_DEVICES"]),
                          self.calls, frame_slot]),)

    def segment_generated_frame(self, *, frame_path, grounding_prompt):
        return self.segment_frame(frame_path=frame_path, frame_slot=0,
                                  grounding_prompt=grounding_prompt)

    def close(self):
        if getattr(self.config, "closed_root", None):
            (self.config.closed_root / str(os.getpid())).touch()
        if getattr(self.config, "close_failure", False):
            raise RuntimeError("close failed")


def fake_factory(config):
    if getattr(config, "startup_failure", False) and os.environ["CUDA_VISIBLE_DEVICES"] == "7":
        raise RuntimeError("startup failed")
    return FakeSegmenter(config)


def make_pool(workers=1, gpus=(3,), *, timeout=5, grace=1, startup_failure=False):
    name = "r2v_data_v2.v3.post_mask_epoch_sa_process_pool"
    assert importlib.util.find_spec(name) is not None, "SA spawn pool is not implemented"
    cls = importlib.import_module(name).SASamProcessPool
    return cls(SimpleNamespace(device="cuda:6", startup_failure=startup_failure),
               pool=WorkerPoolConfig(gpu_ids=gpus, timeout_seconds=timeout,
                                     shutdown_grace_seconds=grace),
               workers_per_gpu=workers, backend_factory=fake_factory)


def call(handle, prompt="ok", path="/private/clip/frame.jpg", slot=4):
    frame = _FRAME_ROOT / str(path).lstrip("/")
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.touch(exist_ok=True)
    return handle.segment_frame(frame_path=frame, frame_slot=slot,
                                grounding_prompt=prompt)[0]


@pytest.mark.parametrize("workers", [1, 2, 4])
def test_spawn_isolation_gpu_mapping_and_serial_model_reuse(monkeypatch, workers):
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.setenv(key, "outer-job")
    pool = make_pool(workers, gpus=(3, 7))
    pool.start()
    try:
        assert pool.slot_count == 2 * workers
        with ThreadPoolExecutor(max_workers=pool.slot_count) as executor:
            results = list(executor.map(lambda slot: call(pool.handle_for_slot(slot), "slow"),
                                        range(pool.slot_count)))
        assert len({int(result[0]) for result in results}) == 2 * workers
        assert sorted(int(result[1]) for result in results) == [3] * workers + [7] * workers
        handle = pool.handle_for_slot(0)
        first = call(handle, path="/private/affinity/frame.jpg")
        second = call(handle, path="/private/affinity/other.jpg")
        assert second[0] == first[0]
        assert second[2] == first[2] + 1
        assert first.flags.owndata
        second[0] = 0
        assert first[0] > 0
        generated_path = _FRAME_ROOT / "generated.png"
        generated_path.touch()
        generated = handle.segment_generated_frame(frame_path=generated_path,
                                                    grounding_prompt="ok")[0]
        assert generated[3] == 0
    finally:
        children = pool._children[:]
        pool.stop()
    assert all(not child.is_alive() for child in children)
    pool.stop()
    assert os.environ["RANK"] == "outer-job"


@pytest.mark.parametrize("generated", [False, True])
@pytest.mark.parametrize("missing", [False, True])
def test_explicit_input_error_keeps_worker_usable(tmp_path, generated, missing):
    pool = make_pool()
    pool.start()
    try:
        frame = tmp_path / "input.jpg"
        if not missing:
            frame.touch()
        handle = pool.handle_for_slot(0)
        method = handle.segment_generated_frame if generated else handle.segment_frame
        kwargs = {} if generated else {"frame_slot": 0}
        with pytest.raises(FileNotFoundError if missing else ValueError):
            method(frame_path=frame, grounding_prompt="ok" if missing else "  ", **kwargs)
        assert call(pool.handle_for_slot(0))[0] > 0
    finally:
        pool.stop()


@pytest.mark.parametrize("error", [PermissionError("denied"), OSError(errno.EIO, "I/O failed")])
def test_input_access_infrastructure_error_is_not_clip_local(monkeypatch, error):
    pool = make_pool()
    pool.start()
    try:
        def fail_access(_path):
            raise error

        monkeypatch.setattr(Path, "is_file", fail_access)
        with pytest.raises(EpochResourceError, match="input access failed"):
            call(pool.handle_for_slot(0))
    finally:
        pool.stop()


@pytest.mark.parametrize("prompt", ["crash", "oom", "permission", "io", "triton",
                                    "type", "value", "missing"])
def test_infrastructure_failure_is_not_a_clip_local_failure(prompt):
    pool = make_pool()
    pool.start()
    try:
        with pytest.raises(EpochResourceError):
            call(pool.handle_for_slot(0), prompt)
    finally:
        children = pool._children[:]
        pool.stop()
    assert all(not child.is_alive() for child in children)


@pytest.mark.parametrize("prompt,error", [("interrupt", KeyboardInterrupt), ("exit", SystemExit)])
def test_fatal_control_flow_crosses_process_boundary(prompt, error):
    pool = make_pool()
    pool.start()
    try:
        with pytest.raises(error):
            call(pool.handle_for_slot(0), prompt)
    finally:
        pool.stop()


def test_partial_startup_failure_reaps_all_owned_children():
    before = {child.pid for child in multiprocessing.active_children()}
    pool = make_pool(gpus=(3, 7), startup_failure=True)
    with pytest.raises(EpochResourceError, match="startup failed"):
        pool.start()
    assert {child.pid for child in multiprocessing.active_children()} <= before
    pool.stop()


def test_hung_child_times_out_and_stop_is_bounded():
    pool = make_pool(timeout=0.15, grace=0.15)
    pool.start()
    children = pool._children[:]
    started = time.monotonic()
    try:
        with pytest.raises(EpochResourceError, match="timed out"):
            call(pool.handle_for_slot(0), "hang")
    finally:
        pool.stop()
    assert time.monotonic() - started < 3
    assert all(not child.is_alive() for child in children)


def test_failed_worker_does_not_discard_successful_sibling():
    pool = make_pool(workers=2)
    pool.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            good = executor.submit(call, pool.handle_for_slot(0), "slow")
            time.sleep(0.05)
            bad = executor.submit(call, pool.handle_for_slot(1), "oom")
            with pytest.raises(EpochResourceError):
                bad.result()
            assert good.result()[0] > 0
    finally:
        pool.stop()


def test_stop_interrupts_a_busy_hung_rpc_without_leaking_threads():
    pool = make_pool(grace=0.1)
    pool.start()
    children = pool._children[:]
    with ThreadPoolExecutor(max_workers=1) as executor:
        busy = executor.submit(call, pool.handle_for_slot(0), "hang")
        deadline = time.monotonic() + 2
        while not pool.counters()["sa_sam_active_peak"]:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        pool.stop()
        with pytest.raises(EpochResourceError):
            busy.result(timeout=2)
    assert all(not child.is_alive() for child in children)


def test_stop_drains_busy_success_and_closes_model_gracefully(tmp_path):
    pool = make_pool()
    pool.config.closed_root = tmp_path
    pool.start()
    with ThreadPoolExecutor(max_workers=1) as executor:
        busy = executor.submit(call, pool.handle_for_slot(0), "slow")
        deadline = time.monotonic() + 2
        while not pool.counters()["sa_sam_active_peak"]:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        pool.stop()
        pid = int(busy.result(timeout=2)[0])
    assert (tmp_path / str(pid)).is_file()


def test_busy_clip_affinity_never_blocks_an_idle_worker():
    pool = make_pool(workers=2)
    pool.start()
    try:
        handle = pool.handle_for_slot(0)
        warm = call(handle)
        with ThreadPoolExecutor(max_workers=2) as executor:
            busy = executor.submit(call, handle, "slow")
            deadline = time.monotonic() + 2
            while not pool._active:
                assert time.monotonic() < deadline
                time.sleep(0.005)
            available = executor.submit(call, handle)
            assert available.result(timeout=1)[0] != warm[0]
            assert busy.result()[0] == warm[0]
        assert pool.counters()["sa_sam_active_peak"] == 2
    finally:
        pool.stop()


def test_broken_pipe_is_an_infrastructure_failure():
    pool = make_pool()
    pool.start()
    try:
        pool._connections[0].close()
        with pytest.raises(EpochResourceError, match="IPC"):
            call(pool.handle_for_slot(0))
    finally:
        pool.stop()


def test_malformed_worker_reply_is_not_a_local_value_error(monkeypatch):
    pool = make_pool()
    pool.start()
    try:
        monkeypatch.setattr(pool, "_receive", lambda *_args: ("local",))
        with pytest.raises(EpochResourceError, match="reply"):
            call(pool.handle_for_slot(0))
    finally:
        pool.stop()


def test_worker_close_failure_is_reported_after_all_children_are_reaped():
    pool = make_pool(workers=2)
    pool.config.close_failure = True
    pool.start()
    children = pool._children[:]
    with pytest.raises(EpochResourceError, match="shutdown"):
        pool.stop()
    assert all(not child.is_alive() for child in children)
    pool.stop()


def test_unreaped_owned_child_is_retained_for_shutdown_retry():
    class OwnedChild:
        pid = 12345
        exitcode = None
        signals = 0

        def is_alive(self):
            return self.exitcode is None

        def join(self, timeout):
            pass

        def terminate(self):
            self.signals += 1

        def kill(self):
            self.signals += 1

    class Connection:
        closed = False

        def send(self, payload):
            assert not self.closed

        def close(self):
            self.closed = True

    pool = make_pool()
    child, connection = OwnedChild(), Connection()
    pool._children = [child]
    pool._connections = [connection]
    pool._available = {0}
    pool._started = True
    with pytest.raises(EpochResourceError, match="unreaped"):
        pool.stop()
    assert pool._children == [child]
    assert connection.closed
    assert child.signals == 2
    child.exitcode = 0
    pool.stop()
    assert pool._children == []
    assert pool._connections == []
    assert child.signals == 2


@pytest.mark.parametrize("workers", [0, 3, 8, True])
def test_only_supported_execution_counts_are_accepted(workers):
    with pytest.raises(ValueError, match="1, 2, 4"):
        make_pool(workers)
