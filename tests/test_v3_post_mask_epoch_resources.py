"""Owned resource lifecycle, true concurrency and resource transitions."""

from __future__ import annotations

import os
import subprocess
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    RESOURCE_BOOGU,
    RESOURCE_QWEN,
    RESOURCE_SAM,
    JobResult,
    ModelJob,
)
from r2v_data_v2.v3.post_mask_epoch_resources import (
    CPU_WORKERS_ENV,
    DEFAULT_CPU_WORKERS,
    HASH_WORKER_CAP,
    EpochResource,
    EpochResourceError,
    OwnedProcess,
    QwenConcurrentExecutor,
    QwenEpochConfig,
    QwenEpochResource,
    ResourceEpochManager,
    SerialBatchExecutor,
    SubprocessEpochProcessManager,
    WorkerEpochResource,
    WorkerPoolConfig,
    WorkerSlotExecutor,
    boogu_worker_factory,
    build_sam_epoch,
    deterministic_slot,
    limit_process_native_threads,
    port_in_use,
    resolve_cpu_workers,
    resolve_hash_workers,
    resolve_qwen_max_inflight,
    served_model_ids,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import ResourceEpochScheduler
from r2v_data_v2.v3.post_mask_epoch_state import GroupLedger

MODEL = Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct")
# Derived, not literal: str(Path(...)) is platform-normalised, and the health
# check compares against str(model_path).
FULL_MODEL_ID = str(MODEL)
BASENAME_MODEL_ID = "Qwen3-VL-32B-Instruct"


class _FakeProcessManager:
    """Records every spawn and every stop; knows nothing about real signals."""

    def __init__(self) -> None:
        self.spawned: list[OwnedProcess] = []
        self.stopped: list[OwnedProcess] = []
        self._alive: dict[int, bool] = {}
        self._next_pid = 1000

    def spawn(self, argv, *, env, log_path=None):
        pid = self._next_pid
        self._next_pid += 1
        process = OwnedProcess(
            pid=pid, pgid=pid + 500, argv=tuple(argv), log_path=log_path
        )
        self.spawned.append(process)
        self._alive[pid] = True
        return process

    def alive(self, process: OwnedProcess) -> bool:
        return self._alive.get(process.pid, False)

    def stop(self, process: OwnedProcess, *, grace_seconds: float) -> bool:
        self.stopped.append(process)
        self._alive[process.pid] = False
        return True


def _qwen_config(**kwargs) -> QwenEpochConfig:
    defaults: dict[str, Any] = {
        "model_path": MODEL,
        "port": 8123,
        "executable": "/usr/local/bin/vllm",
        # Short so a mismatched model id fails fast instead of hanging a suite.
        "startup_timeout_seconds": 5.0,
    }
    defaults.update(kwargs)
    return QwenEpochConfig(**defaults)


def _healthy(*model_ids: str):
    served = list(model_ids) or [FULL_MODEL_ID]

    def probe(base_url: str) -> dict:
        return {"data": [{"id": model_id} for model_id in served]}

    return probe


def _boogu_job(clip_uid: str, **kwargs) -> ModelJob:
    return ModelJob.create(
        job_type=kwargs.pop("job_type", "boogu_edit"),
        resource=RESOURCE_BOOGU,
        canonical_shard="shard-000000000-000000999",
        clip_uid=clip_uid,
        semantic_inputs={"prompt": "x"},
        model_identity="boogu",
        **kwargs,
    )


def _qwen_job(clip_uid: str, **kwargs) -> ModelJob:
    return ModelJob.create(
        job_type=kwargs.pop("job_type", "qwen_judge"),
        resource=RESOURCE_QWEN,
        canonical_shard="shard-000000000-000000999",
        clip_uid=clip_uid,
        semantic_inputs={"prompt": "x"},
        model_identity="qwen",
        **kwargs,
    )


# --------------------------------------------------------------------------
# Qwen epoch
# --------------------------------------------------------------------------


def test_qwen_epoch_starts_once_and_stops_once(tmp_path: Path):
    manager = _FakeProcessManager()
    resource = QwenEpochResource(
        _qwen_config(),
        process_manager=manager,
        log_root=tmp_path,
        health_probe=_healthy(),
    )
    resource.start()
    with pytest.raises(EpochResourceError):
        resource.start()
    resource.stop()
    assert resource.start_count == 1
    assert resource.stop_count == 1
    assert len(manager.spawned) == 1


def test_qwen_epoch_uses_all_eight_gpus_with_tp1_dp8():
    config = _qwen_config()
    assert config.gpu_ids == (0, 1, 2, 3, 4, 5, 6, 7)
    argv = config.argv()
    assert argv[argv.index("--tensor-parallel-size") + 1] == "1"
    assert argv[argv.index("--data-parallel-size") + 1] == "8"
    assert config.environment()["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_qwen_argv_matches_server_verified_invocation():
    config = _qwen_config(
        allowed_local_media_path=Path("/mnt/workspace/public/dataset"),
        served_model_name=FULL_MODEL_ID,
    )
    argv = list(config.argv())
    assert argv[0] == "/usr/local/bin/vllm"
    assert argv[1] == "serve"
    assert str(MODEL) in argv
    for flag, value in (
        ("--host", "127.0.0.1"),
        ("--port", str(config.port)),
        ("--max-model-len", "49152"),
        ("--dtype", "bfloat16"),
        ("--served-model-name", FULL_MODEL_ID),
        ("--allowed-local-media-path", str(Path("/mnt/workspace/public/dataset"))),
    ):
        assert flag in argv
        assert argv[argv.index(flag) + 1] == value


def test_expected_served_model_id_prefers_explicit_name_then_full_path():
    assert _qwen_config().expected_served_model_id == str(MODEL)
    assert (
        _qwen_config(served_model_name="my-alias").expected_served_model_id
        == "my-alias"
    )


def test_health_check_accepts_full_absolute_path_model_id(tmp_path: Path):
    resource = QwenEpochResource(
        _qwen_config(),
        process_manager=_FakeProcessManager(),
        log_root=tmp_path,
        health_probe=_healthy(FULL_MODEL_ID),
    )
    resource.start()
    assert resource.open


def test_health_check_accepts_custom_served_model_name(tmp_path: Path):
    resource = QwenEpochResource(
        _qwen_config(served_model_name="my-alias"),
        process_manager=_FakeProcessManager(),
        log_root=tmp_path,
        health_probe=_healthy("my-alias"),
    )
    resource.start()
    assert resource.open


def test_health_check_scans_all_advertised_models(tmp_path: Path):
    resource = QwenEpochResource(
        _qwen_config(served_model_name="third-model"),
        process_manager=_FakeProcessManager(),
        log_root=tmp_path,
        health_probe=_healthy("first-model", "second-model", "third-model"),
    )
    resource.start()
    assert resource.open


def test_health_check_rejects_basename_only_response(tmp_path: Path):
    """The real server advertises the full path, not the basename."""
    resource = QwenEpochResource(
        _qwen_config(startup_timeout_seconds=0.2),
        process_manager=_FakeProcessManager(),
        log_root=tmp_path,
        health_probe=_healthy(BASENAME_MODEL_ID),
    )
    with pytest.raises(EpochResourceError):
        resource.start()


def test_served_model_ids_helper_handles_empty_and_multiple():
    assert served_model_ids({}) == ()
    assert served_model_ids({"data": [{"id": "a"}, {"id": "b"}]}) == ("a", "b")


def test_qwen_epoch_fails_fast_when_port_is_already_serving(tmp_path: Path):
    import r2v_data_v2.v3.post_mask_epoch_resources as module

    manager = _FakeProcessManager()
    resource = QwenEpochResource(
        _qwen_config(),
        process_manager=manager,
        log_root=tmp_path,
        health_probe=_healthy(),
    )
    original = module.port_in_use
    module.port_in_use = lambda host, port: True
    try:
        with pytest.raises(EpochResourceError, match="already serving"):
            resource.start()
        assert manager.spawned == []
    finally:
        module.port_in_use = original


def test_qwen_shutdown_signals_only_the_owned_process(tmp_path: Path):
    manager = _FakeProcessManager()
    resource = QwenEpochResource(
        _qwen_config(),
        process_manager=manager,
        log_root=tmp_path,
        health_probe=_healthy(),
    )
    resource.start()
    owned = manager.spawned[0]
    external = OwnedProcess(pid=4242, pgid=4243, argv=("external-service",))
    manager._alive[external.pid] = True
    resource.stop()
    assert [p.pid for p in manager.stopped] == [owned.pid]
    assert external.pid not in [p.pid for p in manager.stopped]
    assert manager._alive[external.pid] is True


def test_missing_vllm_executable_fails_without_spawning(tmp_path: Path):
    import r2v_data_v2.v3.post_mask_epoch_resources as module

    manager = _FakeProcessManager()
    resource = QwenEpochResource(
        _qwen_config(executable="definitely-not-vllm"),
        process_manager=manager,
        log_root=tmp_path,
        health_probe=_healthy(),
    )
    original = module.shutil.which
    module.shutil.which = lambda name: None
    try:
        with pytest.raises(EpochResourceError, match="not found on PATH"):
            resource.start()
        assert manager.spawned == []
    finally:
        module.shutil.which = original


def test_port_in_use_helper_is_false_for_a_closed_port():
    assert port_in_use("127.0.0.1", 1) is False


# --------------------------------------------------------------------------
# Popen wait/reap
# --------------------------------------------------------------------------


class _FakePopen:
    instances: ClassVar[list[_FakePopen]] = []

    def __init__(self, argv, **kwargs):
        self.argv = list(argv)
        self.pid = 7000 + len(_FakePopen.instances)
        self.wait_calls = 0
        self.kill_calls = 0
        self._alive = True
        _FakePopen.instances.append(self)

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self.wait_calls += 1
        self._alive = False
        return 0

    def kill(self):
        self.kill_calls += 1
        self._alive = False


def test_real_process_manager_reaps_children(tmp_path, monkeypatch):
    _FakePopen.instances = []
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    manager = SubprocessEpochProcessManager()
    process = manager.spawn(
        ["/usr/local/bin/vllm", "serve"], env={"CUDA_VISIBLE_DEVICES": "0,1"}
    )
    assert manager.open_children == (process.pid,)
    manager.stop(process, grace_seconds=0.1)
    assert process.pid in manager.reaped
    assert manager.open_children == ()
    assert _FakePopen.instances[0].wait_calls >= 1


def test_real_process_manager_reaps_after_startup_failure(tmp_path, monkeypatch):
    _FakePopen.instances = []
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    manager = SubprocessEpochProcessManager()
    process = manager.spawn(["/bin/false"], env={})
    manager.stop(process, grace_seconds=0.1)
    assert process.pid in manager.reaped
    assert manager.open_children == ()


# --------------------------------------------------------------------------
# Worker epoch
# --------------------------------------------------------------------------


class _FakeWorker:
    def __init__(self, slot: int, gpu_id: int):
        self.slot, self.gpu_id = slot, gpu_id
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_image_edit_worker_factory_selects_qwen_for_each_physical_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy resource name owns one Qwen worker per physical GPU."""
    from types import SimpleNamespace

    import r2v_data_v2.v3.qwen_image21_backend as qwen_image_module

    allowed = (tmp_path / "workspace" / "data").resolve()
    python = allowed / "venv" / "python"
    model_path = Path("/mnt/workspace/public/pretrained/Qwen/Qwen-Image-2.1")
    pe_path = Path("/mnt/workspace/public/pretrained/Qwen/Qwen-Image-2.1-PE-I2I")
    config = SimpleNamespace(
        backend="qwen_image_2_1",
        python_executable=python,
        model_path=model_path,
        prompt_enhancer_i2i_path=pe_path,
        num_inference_steps=40,
        timeout_seconds=3600,
    )
    captured: list[Any] = []

    class _Backend:
        def __init__(self, worker_config: Any) -> None:
            captured.append(worker_config)

        def start(self, *, stderr_log_path: Path) -> None:
            pass

    monkeypatch.setattr(qwen_image_module, "QwenImage21SubprocessBackend", _Backend)

    factory = boogu_worker_factory(
        config=config,
        temporary_root=allowed / "tmp",
        allowed_server_root=allowed,
        pool=WorkerPoolConfig(gpu_ids=(1, 3), timeout_seconds=900.0),
    )
    first = factory(0, 1)
    second = factory(1, 3)

    assert isinstance(first, _Backend) and isinstance(second, _Backend)
    assert len(captured) == 2
    assert [item.cuda_visible_devices for item in captured] == ["1", "3"]
    assert all(item.timeout_seconds == 900 for item in captured)
    assert all(type(item.timeout_seconds) is int for item in captured)
    assert all(item.model_path == model_path for item in captured)
    assert all(item.prompt_enhancer_i2i_path == pe_path for item in captured)
    assert all(item.num_inference_steps == 40 for item in captured)


def test_image_edit_worker_factory_selects_boogu_for_default_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import r2v_data_v2.v3.reference_edit_boogu as boogu_module

    allowed = (tmp_path / "workspace" / "data").resolve()
    python = allowed / "venv" / "python"
    code_root = allowed / "vendor" / "Boogu-Image"
    model_path = allowed / "models" / "Boogu"
    config = SimpleNamespace(
        backend="boogu_image_0_1_edit_turbo",
        python_executable=python,
        code_root=code_root,
        model_path=model_path,
        model_revision="revision",
        timeout_seconds=3600,
    )
    captured: list[Any] = []

    class _Backend:
        def __init__(self, worker_config: Any) -> None:
            worker_config.validate()
            captured.append(worker_config)

        def start(self, *, stderr_log_path: Path) -> None:
            pass

    monkeypatch.setattr(boogu_module, "BooguSubprocessBackend", _Backend)

    factory = boogu_worker_factory(
        config=config,
        temporary_root=allowed / "tmp",
        allowed_server_root=allowed,
        pool=WorkerPoolConfig(gpu_ids=(1, 3), timeout_seconds=900.0),
    )
    first = factory(0, 1)
    second = factory(1, 3)

    assert isinstance(first, _Backend) and isinstance(second, _Backend)
    assert [item.cuda_visible_devices for item in captured] == ["1", "3"]
    assert all(item.timeout_seconds == 900 for item in captured)
    assert all(item.model_path == model_path for item in captured)


def test_worker_epoch_loads_one_worker_per_gpu_once(tmp_path: Path):
    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(gpu_ids=(0, 1, 2, 3, 4, 5, 6, 7)),
        process_manager=_FakeProcessManager(),
        worker_factory=_FakeWorker,
        log_root=tmp_path,
    )
    resource.start()
    assert len(resource.workers) == 8
    assert [w.gpu_id for w in resource.workers] == list(range(8))
    assert resource.start_count == 1
    resource.stop()
    assert all(w.closed for w in resource.workers)
    assert resource.stop_count == 1


def test_worker_epoch_refuses_to_serve_before_start(tmp_path: Path):
    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(gpu_ids=(0,)),
        process_manager=_FakeProcessManager(),
        worker_factory=_FakeWorker,
        log_root=tmp_path,
    )
    with pytest.raises(EpochResourceError):
        resource.handle_for(_boogu_job("clip-1"))


def test_deterministic_slot_is_stable_and_in_range():
    job = _boogu_job("clip-1")
    slot = deterministic_slot(job, 8)
    assert 0 <= slot < 8
    assert slot == deterministic_slot(job, 8)


def test_sam_epoch_respects_non_contiguous_gpu_ids(tmp_path: Path):
    built: list[Any] = []

    @dataclass(frozen=True)
    class _FakeSamConfig:
        device: str = "cuda"

    def builder(config):
        built.append(config.device)
        return _FakeWorker(len(built), 0)

    resource = build_sam_epoch(
        _FakeSamConfig(),
        pool=WorkerPoolConfig(gpu_ids=(2, 4)),
        process_manager=_FakeProcessManager(),
        log_root=tmp_path,
        backend_builder=builder,
    )
    resource.start()
    assert built == ["cuda:2", "cuda:4"]


# --------------------------------------------------------------------------
# True concurrency
# --------------------------------------------------------------------------


class _Tracker:
    """Counts concurrent activity and forces overlap with a barrier."""

    def __init__(self, parties: int, *, timeout: float = 10.0):
        self.barrier = threading.Barrier(parties, timeout=timeout)
        self.active = 0
        self.max_active = 0
        self.slot_active: dict[int, int] = {}
        self.slot_max: dict[int, int] = {}
        self.lock = threading.Lock()

    def enter(self, slot: int | None = None) -> None:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if slot is not None:
                self.slot_active[slot] = self.slot_active.get(slot, 0) + 1
                self.slot_max[slot] = max(
                    self.slot_max.get(slot, 0), self.slot_active[slot]
                )

    def exit(self, slot: int | None = None) -> None:
        with self.lock:
            self.active -= 1
            if slot is not None:
                self.slot_active[slot] -= 1


def test_worker_slot_executor_runs_eight_slots_concurrently():
    tracker = _Tracker(parties=8)

    def runner(job: ModelJob, slot: int) -> JobResult:
        tracker.enter(slot)
        try:
            tracker.barrier.wait()
            return JobResult(OUTCOME_COMPLETED, payload={"slot": slot})
        finally:
            tracker.exit(slot)

    jobs = [_boogu_job(f"clip-{i:03d}") for i in range(8)]
    results = WorkerSlotExecutor(runner, slot_count=8).execute_batch(jobs)

    assert len(results) == 8
    assert tracker.max_active == 8
    assert len(tracker.slot_max) == 8
    assert all(value == 1 for value in tracker.slot_max.values())


def test_worker_slot_executor_never_overlaps_a_slot_with_itself():
    tracker = _Tracker(parties=8)

    def runner(job: ModelJob, slot: int) -> JobResult:
        tracker.enter(slot)
        try:
            tracker.barrier.wait()
            return JobResult(OUTCOME_COMPLETED, payload={})
        finally:
            tracker.exit(slot)

    jobs = [_boogu_job(f"clip-{i:03d}") for i in range(16)]
    results = WorkerSlotExecutor(runner, slot_count=8).execute_batch(jobs)

    assert len(results) == 16
    # Two jobs per slot, but never two at once on the same GPU.
    assert len(tracker.slot_max) == 8
    assert all(value == 1 for value in tracker.slot_max.values())
    assert tracker.max_active == 8


def test_worker_slot_executor_isolates_exceptions_per_job():
    def runner(job: ModelJob, slot: int) -> JobResult:
        if job.clip_uid == "clip-bad":
            raise RuntimeError("boom")
        return JobResult(OUTCOME_COMPLETED, payload={})

    jobs = [_boogu_job("clip-ok"), _boogu_job("clip-bad")]
    results = WorkerSlotExecutor(runner, slot_count=8).execute_batch(jobs)
    assert len(results) == 2
    failures = [r for r in results.values() if r.exception is not None]
    assert len(failures) == 1
    assert isinstance(failures[0].exception, RuntimeError)


def test_qwen_executor_sends_requests_concurrently():
    tracker = _Tracker(parties=8)

    def runner(job: ModelJob, endpoint: object) -> JobResult:
        tracker.enter()
        try:
            tracker.barrier.wait()
            return JobResult(OUTCOME_COMPLETED, payload={})
        finally:
            tracker.exit()

    jobs = [_qwen_job(f"clip-{i:03d}") for i in range(8)]
    executor = QwenConcurrentExecutor(
        runner, endpoint="http://127.0.0.1:8000/v1", max_inflight=8
    )
    results = executor.execute_batch(jobs)
    assert len(results) == 8
    assert tracker.max_active == 8


def test_qwen_execution_only_inflight_32_exceeds_dp8(monkeypatch):
    monkeypatch.setenv("POST_MASK_QWEN_MAX_INFLIGHT", "32")
    assert resolve_qwen_max_inflight() == 32
    tracker = _Tracker(parties=32)

    def runner(job: ModelJob, endpoint: object) -> JobResult:
        tracker.enter()
        try:
            tracker.barrier.wait(timeout=10)
            return JobResult(OUTCOME_COMPLETED, payload={})
        finally:
            tracker.exit()

    jobs = [_qwen_job(f"clip-{i:03d}") for i in range(32)]
    ids = [job.job_id() for job in jobs]
    results = QwenConcurrentExecutor(
        runner, max_inflight=resolve_qwen_max_inflight()
    ).execute_batch(jobs)
    assert len(results) == 32
    assert tracker.max_active == 32
    assert [job.job_id() for job in jobs] == ids


def test_qwen_inflight_override_is_not_semantic_config(monkeypatch, tmp_path):
    from tests.test_v3_pair import _config

    config = _config(tmp_path, monkeypatch)
    fingerprint = config.fingerprint()
    monkeypatch.setenv("POST_MASK_QWEN_MAX_INFLIGHT", "8")
    assert resolve_qwen_max_inflight() == 8
    job_id = _qwen_job("clip-identity").job_id()
    monkeypatch.setenv("POST_MASK_QWEN_MAX_INFLIGHT", "64")
    assert resolve_qwen_max_inflight() == 64
    assert config.fingerprint() == fingerprint
    assert _qwen_job("clip-identity").job_id() == job_id
    monkeypatch.setenv("POST_MASK_QWEN_MAX_INFLIGHT", "0")
    with pytest.raises(EpochResourceError, match="POST_MASK_QWEN_MAX_INFLIGHT"):
        resolve_qwen_max_inflight()


@pytest.mark.parametrize("resource", [RESOURCE_SAM, RESOURCE_BOOGU])
def test_eight_gpu_slots_refill_from_thirty_two_ready_jobs(resource):
    first_wave_ready = threading.Event()
    release_slow = threading.Event()
    ninth_started = threading.Event()
    lock = threading.Lock()
    active = 0
    peak = 0
    started = 0
    slots: set[int] = set()

    def runner(job: ModelJob, slot: int) -> JobResult:
        nonlocal active, peak, started
        with lock:
            active += 1
            started += 1
            peak = max(peak, active)
            slots.add(slot)
            if started == 8:
                first_wave_ready.set()
            if job.clip_uid == "clip-08":
                ninth_started.set()
        try:
            if job.clip_uid == "clip-00":
                assert first_wave_ready.wait(5)
            elif job.clip_uid in {f"clip-{i:02d}" for i in range(1, 8)}:
                assert release_slow.wait(5)
            return JobResult(OUTCOME_COMPLETED, payload={})
        finally:
            with lock:
                active -= 1

    jobs = [
        ModelJob.create(
            job_type="probe", resource=resource,
            canonical_shard="shard-000000000-000000999",
            clip_uid=f"clip-{i:02d}", semantic_inputs={"x": 1},
            model_identity="fake",
        )
        for i in range(32)
    ]
    executor = WorkerSlotExecutor(runner, slot_count=8)
    try:
        for job in jobs[:8]:
            executor.submit(job)
        assert first_wave_ready.wait(5)
        first = executor.collect()
        assert [item.job.clip_uid for item in first] == ["clip-00"]
        executor.submit(jobs[8])
        assert ninth_started.wait(5), "free slot must refill before slow siblings"
        release_slow.set()
        completed = 1
        next_index = 9
        while completed < len(jobs):
            results = executor.collect()
            completed += len(results)
            for _ in results:
                if next_index < len(jobs):
                    executor.submit(jobs[next_index])
                    next_index += 1
        assert peak == 8
        assert slots == set(range(8))
    finally:
        release_slow.set()
        executor.close()


def test_qwen_executor_respects_max_inflight():
    tracker = _Tracker(parties=2)

    def runner(job: ModelJob, endpoint: object) -> JobResult:
        tracker.enter()
        try:
            tracker.barrier.wait()
            return JobResult(OUTCOME_COMPLETED, payload={})
        finally:
            tracker.exit()

    jobs = [_qwen_job(f"clip-{i:03d}") for i in range(6)]
    results = QwenConcurrentExecutor(runner, max_inflight=2).execute_batch(jobs)
    assert len(results) == 6
    assert tracker.max_active <= 2


def test_serial_batch_executor_runs_one_at_a_time():
    tracker = _Tracker(parties=1)
    seen: list[str] = []

    def runner(job: ModelJob, handle: object) -> JobResult:
        tracker.enter()
        seen.append(job.clip_uid)
        tracker.exit()
        return JobResult(OUTCOME_COMPLETED, payload={})

    jobs = [_boogu_job("clip-b"), _boogu_job("clip-a")]
    results = SerialBatchExecutor(runner).execute_batch(jobs)
    assert len(results) == 2
    assert tracker.max_active == 1
    assert seen == ["clip-a", "clip-b"]


# --------------------------------------------------------------------------
# ResourceEpochManager transitions
# --------------------------------------------------------------------------


class _TrackedResource(EpochResource):
    def _start(self) -> None:
        return None

    def _stop(self) -> None:
        return None


def _factory(name: str, executor, log: list[str]):
    def build():
        log.append(f"build:{name}")
        return _TrackedResource(name, process_manager=_FakeProcessManager()), executor

    return build


def test_manager_keeps_at_most_one_resource_open():
    log: list[str] = []
    manager = ResourceEpochManager(
        {
            RESOURCE_QWEN: _factory(
                RESOURCE_QWEN,
                SerialBatchExecutor(lambda j, h: JobResult(OUTCOME_COMPLETED)),
                log,
            ),
            RESOURCE_BOOGU: _factory(
                RESOURCE_BOOGU,
                SerialBatchExecutor(lambda j, h: JobResult(OUTCOME_COMPLETED)),
                log,
            ),
        }
    )
    manager.enter(RESOURCE_QWEN)
    assert manager.open_count == 1
    manager.enter(RESOURCE_BOOGU)
    assert manager.open_count == 1
    assert manager.timeline == [
        ("start", RESOURCE_QWEN),
        ("stop", RESOURCE_QWEN),
        ("start", RESOURCE_BOOGU),
    ]
    manager.close()
    assert manager.open_count == 0
    assert manager.timeline[-1] == ("stop", RESOURCE_BOOGU)
    assert manager.max_open_observed == 1


def test_manager_closes_executor_before_unloading_resource():
    events: list[str] = []

    class _Executor:
        def close(self) -> None:
            events.append("executor_close")

    class _Resource(EpochResource):
        def _start(self) -> None:
            return None

        def _stop(self) -> None:
            events.append("resource_stop")

    manager = ResourceEpochManager(
        {
            RESOURCE_SAM: lambda: (
                _Resource(RESOURCE_SAM, process_manager=_FakeProcessManager()),
                _Executor(),
            )
        }
    )
    manager.enter(RESOURCE_SAM)
    manager.close()

    assert events == ["executor_close", "resource_stop"]


def test_manager_reuses_an_open_resource():
    manager = ResourceEpochManager(
        {
            RESOURCE_SAM: _factory(
                RESOURCE_SAM,
                SerialBatchExecutor(lambda j, h: JobResult(OUTCOME_COMPLETED)),
                [],
            )
        }
    )
    manager.enter(RESOURCE_SAM)
    manager.enter(RESOURCE_SAM)
    assert manager.timeline == [("start", RESOURCE_SAM)]


def test_manager_rejects_unknown_resource():
    with pytest.raises(ValueError):
        ResourceEpochManager({"flux": lambda: (None, None)})


def test_scheduler_drives_resource_transitions_and_fixed_point(tmp_path: Path):
    boogu_executor = SerialBatchExecutor(
        lambda job, handle: JobResult(OUTCOME_COMPLETED, payload={"ok": True})
    )
    qwen_executor = SerialBatchExecutor(
        lambda job, handle: JobResult(OUTCOME_COMPLETED, payload={"ok": True})
    )
    manager = ResourceEpochManager(
        {
            RESOURCE_BOOGU: _factory(RESOURCE_BOOGU, boogu_executor, []),
            RESOURCE_QWEN: _factory(RESOURCE_QWEN, qwen_executor, []),
        }
    )
    first = _boogu_job("clip-001")
    unlocked = _qwen_job("clip-002", job_type="removal_judge")
    second_boogu = _boogu_job(
        "clip-003", job_type="removal_candidate2", attempt_index=1, seed=2
    )

    def finalize(job: ModelJob, result: JobResult):
        if job.job_id() == first.job_id():
            return (unlocked,)
        if job.job_id() == unlocked.job_id():
            return (second_boogu,)
        return ()

    scheduler = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "group"),
        finalize=finalize,
        resource_manager=manager,
    )
    outcome = scheduler.run([first])
    assert outcome["completed"] is True
    assert manager.open_count == 0
    assert [f"{a}:{n}" for a, n in manager.timeline] == [
        "start:boogu",
        "stop:boogu",
        "start:qwen",
        "stop:qwen",
        "start:boogu",
        "stop:boogu",
    ]
    assert manager.max_open_observed == 1


def test_scheduler_unloads_resource_when_executor_raises(tmp_path: Path):
    class _Exploding:
        def execute_batch(self, jobs):
            raise RuntimeError("executor blew up")

    manager = ResourceEpochManager(
        {RESOURCE_BOOGU: _factory(RESOURCE_BOOGU, _Exploding(), [])}
    )
    scheduler = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "group"),
        finalize=lambda job, result: (),
        resource_manager=manager,
    )
    with pytest.raises(RuntimeError, match="executor blew up"):
        scheduler.run([_boogu_job("clip-001")])
    assert manager.open_count == 0
    assert manager.timeline[-1] == ("stop", RESOURCE_BOOGU)


def test_scheduler_records_manager_counters(tmp_path: Path):
    manager = ResourceEpochManager(
        {
            RESOURCE_BOOGU: _factory(
                RESOURCE_BOOGU,
                SerialBatchExecutor(lambda j, h: JobResult(OUTCOME_COMPLETED)),
                [],
            )
        }
    )
    scheduler = ResourceEpochScheduler(
        ledger=GroupLedger(tmp_path / "group"),
        finalize=lambda job, result: (),
        resource_manager=manager,
    )
    outcome = scheduler.run([_boogu_job("clip-001")])
    lifecycle = outcome["resource_lifecycle"]
    assert lifecycle["timeline"] == ["start:boogu", "stop:boogu"]
    assert lifecycle["max_open_observed"] == 1
    # Lifecycle counters are separate from model-job counters.
    assert "_manager" not in outcome["diagnostics"]["resources"]
    assert outcome["diagnostics"]["resource_lifecycle"]["timeline"] == [
        "start:boogu",
        "stop:boogu",
    ]


def test_lifecycle_counters_survive_unload_and_accumulate(tmp_path: Path):
    manager = ResourceEpochManager(
        {
            RESOURCE_BOOGU: _factory(
                RESOURCE_BOOGU,
                SerialBatchExecutor(lambda j, h: JobResult(OUTCOME_COMPLETED)),
                [],
            ),
            RESOURCE_QWEN: _factory(
                RESOURCE_QWEN,
                SerialBatchExecutor(lambda j, h: JobResult(OUTCOME_COMPLETED)),
                [],
            ),
        }
    )
    manager.enter(RESOURCE_QWEN)
    manager.enter(RESOURCE_BOOGU)
    manager.enter(RESOURCE_QWEN)
    manager.close()
    counters = manager.counters()
    # Qwen was entered twice; its counters must aggregate both epochs.
    assert counters["resources"][RESOURCE_QWEN]["start_count"] == 2
    assert counters["resources"][RESOURCE_QWEN]["stop_count"] == 2
    assert counters["resources"][RESOURCE_BOOGU]["start_count"] == 1
    assert counters["resources"][RESOURCE_BOOGU]["stop_count"] == 1
    assert counters["timeline"] == [
        "start:qwen",
        "stop:qwen",
        "start:boogu",
        "stop:boogu",
        "start:qwen",
        "stop:qwen",
    ]


# --------------------------------------------------------------------------
# Process ownership safety
# --------------------------------------------------------------------------


class _RecordingKillpg:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, pgid: int, signal_number: int) -> None:
        self.calls.append((pgid, signal_number))
        raise ProcessLookupError("no such process group")


def test_stop_never_signals_an_unowned_process(monkeypatch):
    recorder = _RecordingKillpg()
    monkeypatch.setattr(os, "killpg", recorder, raising=False)
    manager = SubprocessEpochProcessManager()
    stranger = OwnedProcess(pid=9999, pgid=9999, argv=("someone-elses-service",))
    assert manager.stop(stranger, grace_seconds=0.1) is False
    assert recorder.calls == []
    assert manager.reaped == []


def test_stop_reaps_already_exited_child_without_signalling(monkeypatch):
    _FakePopen.instances = []
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    recorder = _RecordingKillpg()
    monkeypatch.setattr(os, "killpg", recorder, raising=False)

    manager = SubprocessEpochProcessManager()
    process = manager.spawn(["/bin/true"], env={})
    # The child already exited (for example a failed health check).
    _FakePopen.instances[0]._alive = False

    assert manager.stop(process, grace_seconds=0.1) is True
    assert recorder.calls == []
    assert process.pid in manager.reaped
    assert manager.open_children == ()


def test_stop_signals_only_a_live_owned_child(monkeypatch):
    _FakePopen.instances = []
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    recorder = _RecordingKillpg()
    monkeypatch.setattr(os, "killpg", recorder, raising=False)

    manager = SubprocessEpochProcessManager()
    process = manager.spawn(["/usr/local/bin/vllm", "serve"], env={})
    assert manager.stop(process, grace_seconds=0.1) is True
    # Every signal targets the owned pgid: TERM first, then KILL only because
    # the fake child ignores signals and stays alive.
    assert recorder.calls
    assert {call[0] for call in recorder.calls} == {process.pgid}
    assert len(recorder.calls) <= 2
    assert process.pid in manager.reaped


# --------------------------------------------------------------------------
# Worker handle API and parallel startup
# --------------------------------------------------------------------------


def test_worker_slot_executor_hands_backend_to_runner(tmp_path: Path):
    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(),
        process_manager=_FakeProcessManager(),
        worker_factory=_FakeWorker,
        log_root=tmp_path,
    )
    resource.start()
    seen: list[Any] = []

    def runner(job: ModelJob, handle: Any) -> JobResult:
        seen.append(handle)
        return JobResult(OUTCOME_COMPLETED, payload={})

    jobs = [_boogu_job(f"clip-{i:03d}") for i in range(8)]
    WorkerSlotExecutor(runner, slot_count=8, resource=resource).execute_batch(jobs)
    # The semantic runner sees backend handles, never GPU slot ids.
    assert all(isinstance(handle, _FakeWorker) for handle in seen)
    assert sorted(handle.gpu_id for handle in seen) == list(range(8))


def test_worker_slot_executor_keeps_one_job_per_handle_at_a_time(tmp_path: Path):
    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(),
        process_manager=_FakeProcessManager(),
        worker_factory=_FakeWorker,
        log_root=tmp_path,
    )
    resource.start()
    tracker = _Tracker(parties=8)

    def runner(job: ModelJob, handle: Any) -> JobResult:
        slot = handle.slot
        tracker.enter(slot)
        try:
            tracker.barrier.wait()
            return JobResult(OUTCOME_COMPLETED, payload={})
        finally:
            tracker.exit(slot)

    jobs = [_boogu_job(f"clip-{i:03d}") for i in range(16)]
    WorkerSlotExecutor(runner, slot_count=8, resource=resource).execute_batch(jobs)
    assert tracker.max_active == 8
    assert len(tracker.slot_max) == 8
    assert all(value == 1 for value in tracker.slot_max.values())


def test_worker_startup_loads_all_slots_concurrently(tmp_path: Path):
    tracker = _Tracker(parties=8)

    def factory(slot: int, gpu_id: int) -> _FakeWorker:
        tracker.enter(slot)
        try:
            tracker.barrier.wait()
            return _FakeWorker(slot, gpu_id)
        finally:
            tracker.exit(slot)

    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(),
        process_manager=_FakeProcessManager(),
        worker_factory=factory,
        log_root=tmp_path,
    )
    resource.start()
    assert tracker.max_active == 8
    # Completion order must not change slot mapping.
    assert [worker.gpu_id for worker in resource.workers] == list(range(8))


def test_worker_startup_failure_rolls_back_created_workers(tmp_path: Path):
    created: list[_FakeWorker] = []

    def factory(slot: int, gpu_id: int) -> _FakeWorker:
        if slot == 3:
            raise RuntimeError("boom")
        worker = _FakeWorker(slot, gpu_id)
        created.append(worker)
        return worker

    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(gpu_ids=(0, 1, 2, 3, 4, 5, 6, 7)),
        process_manager=_FakeProcessManager(),
        worker_factory=factory,
        log_root=tmp_path,
    )
    with pytest.raises(EpochResourceError, match="slot 3"):
        resource.start()
    assert created
    assert all(worker.closed for worker in created)
    assert resource.workers == []
    assert resource.open is False


class _CloseCounterWorker:
    """A worker whose ``close()`` may raise, counting every attempt."""

    def __init__(self, slot: int, gpu_id: int, *, close_error: str | None = None):
        self.slot, self.gpu_id = slot, gpu_id
        self._close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise RuntimeError(self._close_error)


def test_partial_startup_rollback_closes_every_built_worker(tmp_path: Path):
    """A failing cleanup must not strand the remaining slots' GPU models."""
    created: dict[int, _CloseCounterWorker] = {}

    def factory(slot: int, gpu_id: int) -> _CloseCounterWorker:
        if slot == 2:
            raise RuntimeError("boom-slot-2")
        worker = _CloseCounterWorker(
            slot, gpu_id, close_error="close failed" if slot == 0 else None
        )
        created[slot] = worker
        return worker

    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(gpu_ids=(0, 1, 2)),
        process_manager=_FakeProcessManager(),
        worker_factory=factory,
        log_root=tmp_path,
    )

    with pytest.raises(EpochResourceError) as excinfo:
        resource.start()

    # slot0's close raised, but slot1 was still asked to close.
    assert created[0].close_calls == 1
    assert created[1].close_calls == 1
    assert 2 not in created
    # The original startup failure stays the primary error ...
    message = str(excinfo.value)
    assert "slot 2" in message
    assert "boom-slot-2" in message
    # ... and the cleanup failure is retained as a chained diagnostic.
    assert "rollback incomplete" in message
    assert "slot 0: close failed" in message
    assert resource.workers == []
    assert resource.open is False


def test_partial_startup_rollback_attempts_every_close_even_when_all_fail(
    tmp_path: Path,
):
    created: dict[int, _CloseCounterWorker] = {}

    def factory(slot: int, gpu_id: int) -> _CloseCounterWorker:
        if slot == 3:
            raise RuntimeError("boom-slot-3")
        worker = _CloseCounterWorker(slot, gpu_id, close_error=f"close-{slot}")
        created[slot] = worker
        return worker

    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(gpu_ids=(0, 1, 2, 3)),
        process_manager=_FakeProcessManager(),
        worker_factory=factory,
        log_root=tmp_path,
    )

    with pytest.raises(EpochResourceError) as excinfo:
        resource.start()

    assert all(worker.close_calls == 1 for worker in created.values())
    message = str(excinfo.value)
    assert "slot 3" in message
    for slot in sorted(created):
        assert f"slot {slot}: close-{slot}" in message
    assert resource.workers == []


def test_handle_for_slot_rejects_out_of_range(tmp_path: Path):
    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(gpu_ids=(0, 1)),
        process_manager=_FakeProcessManager(),
        worker_factory=_FakeWorker,
        log_root=tmp_path,
    )
    resource.start()
    with pytest.raises(EpochResourceError):
        resource.handle_for_slot(5)


# --------------------------------------------------------------------------
# Shutdown safety
# --------------------------------------------------------------------------


class _ClosableWorker:
    def __init__(self, slot: int, gpu_id: int, *, fail: bool = False):
        self.slot, self.gpu_id = slot, gpu_id
        self.closed = False
        self.close_calls = 0
        self.fail = fail

    def close(self) -> None:
        self.close_calls += 1
        if self.fail:
            raise RuntimeError(f"close failed for slot {self.slot}")
        self.closed = True


def test_worker_shutdown_closes_every_worker_even_on_failure(tmp_path: Path):
    created: list[_ClosableWorker] = []

    def factory(slot: int, gpu_id: int) -> _ClosableWorker:
        worker = _ClosableWorker(slot, gpu_id, fail=(slot == 2))
        created.append(worker)
        return worker

    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(),
        process_manager=_FakeProcessManager(),
        worker_factory=factory,
        log_root=tmp_path,
    )
    resource.start()
    assert len(created) == 8
    with pytest.raises(EpochResourceError, match="slot 2"):
        resource.stop()
    # Every worker was asked to close, not just the ones before the failure.
    assert all(worker.close_calls == 1 for worker in created)
    assert resource.workers == []


def test_failed_resource_stop_blocks_the_next_resource():
    started: list[str] = []

    class _BadResource(EpochResource):
        def _start(self) -> None:
            started.append(self.name)

        def _stop(self) -> None:
            raise RuntimeError("stop failed")

    def bad_factory():
        return _BadResource(RESOURCE_QWEN, process_manager=_FakeProcessManager()), SerialBatchExecutor(
            lambda j, h: JobResult(OUTCOME_COMPLETED)
        )

    def next_factory():
        started.append(RESOURCE_BOOGU)
        return _TrackedResource(
            RESOURCE_BOOGU, process_manager=_FakeProcessManager()
        ), SerialBatchExecutor(lambda j, h: JobResult(OUTCOME_COMPLETED))

    manager = ResourceEpochManager(
        {RESOURCE_QWEN: bad_factory, RESOURCE_BOOGU: next_factory}
    )
    manager.enter(RESOURCE_QWEN)
    with pytest.raises(RuntimeError, match="stop failed"):
        manager.enter(RESOURCE_BOOGU)
    # Boogu must never be loaded after a failed Qwen unload.
    assert RESOURCE_BOOGU not in started
    assert manager.open_count == 0


# --------------------------------------------------------------------------
# Control-flow exceptions must propagate
# --------------------------------------------------------------------------


def _system_exit_runner(job: ModelJob, handle: Any) -> JobResult:
    raise SystemExit("interrupted")


def test_serial_executor_propagates_system_exit():
    with pytest.raises(SystemExit):
        SerialBatchExecutor(_system_exit_runner).execute_batch([_boogu_job("clip-1")])


def test_qwen_executor_propagates_system_exit():
    executor = QwenConcurrentExecutor(_system_exit_runner, max_inflight=2)
    with pytest.raises(SystemExit):
        executor.execute_batch([_qwen_job("clip-1"), _qwen_job("clip-2")])


def test_qwen_close_waits_for_running_request():
    """Closing an aborted epoch must not unload Qwen under a live sibling."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    closed = threading.Event()

    def runner(job: ModelJob, endpoint: Any) -> JobResult:
        started.set()
        release.wait(timeout=5)
        finished.set()
        return JobResult(OUTCOME_COMPLETED, payload={})

    executor = QwenConcurrentExecutor(runner, max_inflight=1)
    executor.submit(_qwen_job("clip-slow"))
    assert started.wait(timeout=5)

    def close_executor() -> None:
        executor.close()
        closed.set()

    closer = threading.Thread(target=close_executor, daemon=True)
    closer.start()
    try:
        assert not closed.wait(timeout=0.05), "close returned while Qwen was running"
    finally:
        release.set()
    assert closed.wait(timeout=5)
    closer.join(timeout=5)
    assert finished.is_set()


@pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning"
)
def test_worker_slot_executor_propagates_system_exit():
    # SystemExit is raised inside the slot thread (proving it is not swallowed
    # into a JobExecution) and re-raised on the caller's thread after join.
    executor = WorkerSlotExecutor(_system_exit_runner, slot_count=2)
    with pytest.raises(SystemExit):
        executor.execute_batch([_boogu_job("clip-1")])


def test_worker_slot_close_waits_for_running_job():
    """Closing an aborted epoch must not unload a slot backend mid-call."""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    closed = threading.Event()

    def runner(job: ModelJob, slot: int) -> JobResult:
        started.set()
        release.wait(timeout=5)
        finished.set()
        return JobResult(OUTCOME_COMPLETED, payload={})

    executor = WorkerSlotExecutor(runner, slot_count=1)
    executor.submit(_boogu_job("clip-slow"))
    assert started.wait(timeout=5)

    def close_executor() -> None:
        executor.close()
        closed.set()

    closer = threading.Thread(target=close_executor, daemon=True)
    closer.start()
    try:
        assert not closed.wait(timeout=0.05), "close returned while slot work was running"
    finally:
        release.set()
    assert closed.wait(timeout=5)
    closer.join(timeout=5)
    assert finished.is_set()


# --------------------------------------------------------------------------
# Completion-driven executors report work as it lands
# --------------------------------------------------------------------------


def test_qwen_executor_reports_a_completion_before_slow_siblings_settle():
    """The first result is observable without waiting for the whole batch."""
    both_running = threading.Barrier(2)
    release_slow = threading.Event()

    def runner(job: ModelJob, endpoint: Any) -> JobResult:
        both_running.wait(timeout=5)
        if job.clip_uid == "clip-slow":
            release_slow.wait(timeout=5)
        return JobResult(OUTCOME_COMPLETED, payload={})

    jobs = [_qwen_job("clip-slow"), _qwen_job("clip-fast")]
    executor = QwenConcurrentExecutor(runner, endpoint=None, max_inflight=2)
    try:
        for job in jobs:
            executor.submit(job)
        first = executor.collect()
        assert [e.job.clip_uid for e in first] == ["clip-fast"]
        # The slow sibling is genuinely still in flight at this point.
        release_slow.set()
        assert [e.job.clip_uid for e in executor.collect()] == ["clip-slow"]
    finally:
        executor.close()


def test_worker_slot_executor_refills_a_slot_while_another_slot_is_busy():
    """A freed slot takes the next job without waiting for the other slots."""
    both_running = threading.Barrier(2)
    release_slot1 = threading.Event()
    slot_of: dict[str, int] = {}
    second_started = threading.Event()

    def runner(job: ModelJob, slot: int) -> JobResult:
        slot_of[job.clip_uid] = slot
        if job.clip_uid == "clip-000000":
            both_running.wait(timeout=5)
            return JobResult(OUTCOME_COMPLETED, payload={})
        if job.clip_uid == "clip-000001":
            both_running.wait(timeout=5)
            release_slot1.wait(timeout=5)
            return JobResult(OUTCOME_COMPLETED, payload={})
        # The third job: only reachable once slot 0 is free again.
        second_started.set()
        return JobResult(OUTCOME_COMPLETED, payload={})

    executor = WorkerSlotExecutor(runner, slot_count=2)
    try:
        executor.submit(_boogu_job("clip-000000"))
        executor.submit(_boogu_job("clip-000001"))
        first = executor.collect()
        assert [e.job.clip_uid for e in first] == ["clip-000000"]

        # Slot 1 is still busy on clip-000001; slot 0 must take this job now.
        executor.submit(_boogu_job("clip-000002"))
        assert second_started.wait(timeout=5), "slot 0 did not take the next job"
        release_slot1.set()
        rest: list[Any] = []
        while len(rest) < 2:
            rest.extend(executor.collect())
        assert {e.job.clip_uid for e in rest} == {"clip-000001", "clip-000002"}
        assert slot_of == {
            "clip-000000": 0,
            "clip-000001": 1,
            "clip-000002": 0,
        }
    finally:
        executor.close()


def test_worker_slot_executor_refills_whichever_slot_finishes_first():
    """The slot that frees up takes the next job, not a fixed round-robin slot.

    Slot 0 stays busy on a slow job while slot 1 finishes. The next job must go
    to the slot that is actually free - handing it to a fixed next-in-turn slot
    would park it behind the slow job and leave slot 1 idle, which is exactly the
    stall the streaming refill has to avoid.
    """
    both_running = threading.Barrier(2)
    release_slot0 = threading.Event()
    third_started = threading.Event()
    slot_of: dict[str, int] = {}

    def runner(job: ModelJob, slot: int) -> JobResult:
        slot_of[job.clip_uid] = slot
        if job.clip_uid == "clip-000000":
            # Slow: holds its slot until the very end.
            both_running.wait(timeout=5)
            release_slot0.wait(timeout=5)
            return JobResult(OUTCOME_COMPLETED, payload={})
        if job.clip_uid == "clip-000001":
            # Fast: frees its slot first, out of submission order.
            both_running.wait(timeout=5)
            return JobResult(OUTCOME_COMPLETED, payload={})
        third_started.set()
        return JobResult(OUTCOME_COMPLETED, payload={})

    executor = WorkerSlotExecutor(runner, slot_count=2)
    try:
        executor.submit(_boogu_job("clip-000000"))
        executor.submit(_boogu_job("clip-000001"))
        first = executor.collect()
        assert [e.job.clip_uid for e in first] == ["clip-000001"]

        freed_slot = slot_of["clip-000001"]
        busy_slot = slot_of["clip-000000"]
        assert freed_slot != busy_slot

        executor.submit(_boogu_job("clip-000002"))
        assert third_started.wait(timeout=5), "no free slot took the next job"
        assert slot_of["clip-000002"] == freed_slot, "the freed slot must take it"
        assert slot_of["clip-000002"] != busy_slot

        release_slot0.set()
        rest: list[Any] = []
        while len(rest) < 2:
            rest.extend(executor.collect())
        assert {e.job.clip_uid for e in rest} == {"clip-000000", "clip-000002"}
        assert sorted(slot_of.values()) == sorted((freed_slot, freed_slot, busy_slot))
    finally:
        executor.close()


# --------------------------------------------------------------------------
# Execution-only CPU worker budget
# --------------------------------------------------------------------------


def test_resolve_cpu_workers_precedence(tmp_path: Path, monkeypatch) -> None:
    from tests.test_v3_pair import _config

    config = _config(tmp_path, monkeypatch)
    monkeypatch.delenv(CPU_WORKERS_ENV, raising=False)
    # Compatibility fallback for callers that never set the environment.
    assert resolve_cpu_workers(config) == config.runtime.cpu_workers
    assert resolve_cpu_workers(None) == DEFAULT_CPU_WORKERS

    monkeypatch.setenv(CPU_WORKERS_ENV, "32")
    assert resolve_cpu_workers(config) == 32
    assert resolve_cpu_workers(None) == 32


def test_resolve_cpu_workers_rejects_bad_values(monkeypatch) -> None:
    for raw in ("0", "-1", "many", "1.5"):
        monkeypatch.setenv(CPU_WORKERS_ENV, raw)
        with pytest.raises(EpochResourceError):
            resolve_cpu_workers(None)


def test_cpu_workers_env_never_changes_the_config_fingerprint(
    tmp_path: Path, monkeypatch
) -> None:
    """The worker budget is execution-only, so run identity must not move."""
    from tests.test_v3_pair import _config

    config = _config(tmp_path, monkeypatch)
    before = config.fingerprint()
    monkeypatch.setenv(CPU_WORKERS_ENV, "32")

    assert resolve_cpu_workers(config) == 32
    assert config.fingerprint() == before


def test_hash_worker_budget_is_capped(monkeypatch) -> None:
    monkeypatch.setenv(CPU_WORKERS_ENV, "100")
    assert resolve_hash_workers(None) == HASH_WORKER_CAP
    assert resolve_hash_workers(None, tasks=3) == 3
    assert resolve_hash_workers(None, tasks=0) == 1


def test_limit_process_native_threads_never_raises(monkeypatch) -> None:
    """Containment is best-effort: a missing OpenCV must not fail a run."""
    import sys

    monkeypatch.setitem(sys.modules, "cv2", None)
    limit_process_native_threads()

    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(setNumThreads=1))
    limit_process_native_threads()

    def exploding(_value: int) -> None:
        raise RuntimeError("cv2 refused")

    monkeypatch.setitem(
        sys.modules, "cv2", types.SimpleNamespace(setNumThreads=exploding)
    )
    limit_process_native_threads()
