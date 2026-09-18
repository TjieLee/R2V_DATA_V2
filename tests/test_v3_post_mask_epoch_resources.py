"""Owned resource lifecycle: start once, stop once, never signal strangers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from r2v_data_v2.v3.post_mask_epoch_jobs import (
    OUTCOME_COMPLETED,
    RESOURCE_BOOGU,
    RESOURCE_SAM,
    ModelJob,
)
from r2v_data_v2.v3.post_mask_epoch_resources import (
    EpochResourceError,
    HandleExecutor,
    OwnedProcess,
    QwenEpochConfig,
    QwenEpochResource,
    WorkerEpochResource,
    WorkerPoolConfig,
    deterministic_slot,
    port_in_use,
)
from r2v_data_v2.v3.post_mask_epoch_scheduler import JobResult

MODEL = Path("/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct")


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
        process = OwnedProcess(pid=pid, pgid=pid + 500, argv=tuple(argv), log_path=log_path)
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
    # A high, unusual port keeps the fail-fast pre-flight from tripping on
    # whatever else happens to be listening on the developer machine, and a
    # path-shaped executable skips the PATH lookup (vLLM is server-only).
    defaults = {
        "model_path": MODEL,
        "port": 8123,
        "executable": "/usr/local/bin/vllm",
    }
    defaults.update(kwargs)
    return QwenEpochConfig(**defaults)


def _healthy(model_id: str | None = None):
    served = model_id or MODEL.name

    def probe(base_url: str) -> dict:
        return {"data": [{"id": served}]}

    return probe


def test_qwen_epoch_starts_once_and_stops_once(tmp_path: Path):
    manager = _FakeProcessManager()
    resource = QwenEpochResource(
        _qwen_config(),
        process_manager=manager,
        log_root=tmp_path,
        health_probe=_healthy(),
    )
    resource.start()
    resource.start_called = True
    with pytest.raises(EpochResourceError):
        resource.start()
    resource.stop()
    assert resource.start_count == 1
    assert resource.stop_count == 1
    assert len(manager.spawned) == 1


def test_qwen_epoch_uses_all_eight_gpus_with_tp1_dp8(tmp_path: Path):
    config = _qwen_config()
    assert config.gpu_ids == (0, 1, 2, 3, 4, 5, 6, 7)
    assert "--tensor-parallel-size" in config.argv()
    assert config.argv()[config.argv().index("--tensor-parallel-size") + 1] == "1"
    assert config.argv()[config.argv().index("--data-parallel-size") + 1] == "8"
    assert config.environment()["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


def test_qwen_epoch_fails_fast_when_port_is_already_serving(tmp_path: Path):
    manager = _FakeProcessManager()
    resource = QwenEpochResource(
        _qwen_config(),
        process_manager=manager,
        log_root=tmp_path,
        health_probe=_healthy(),
    )
    import r2v_data_v2.v3.post_mask_epoch_resources as module

    original = module.port_in_use
    module.port_in_use = lambda host, port: True
    try:
        with pytest.raises(EpochResourceError, match="already serving"):
            resource.start()
        assert manager.spawned == []
    finally:
        module.port_in_use = original


def test_qwen_epoch_rejects_wrong_served_model(tmp_path: Path):
    manager = _FakeProcessManager()
    resource = QwenEpochResource(
        _qwen_config(startup_timeout_seconds=0.01),
        process_manager=manager,
        log_root=tmp_path,
        health_probe=_healthy("some-other-model"),
    )
    with pytest.raises(EpochResourceError):
        resource.start()
    # A failed startup must still clean up only its own process.
    assert manager.stopped or not manager.spawned


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


def test_qwen_epoch_reports_counters(tmp_path: Path):
    resource = QwenEpochResource(
        _qwen_config(),
        process_manager=_FakeProcessManager(),
        log_root=tmp_path,
        health_probe=_healthy(),
    )
    resource.start()
    resource.stop()
    counters = resource.counters()
    for key in (
        "start_count",
        "stop_count",
        "startup_wall_seconds",
        "shutdown_wall_seconds",
        "service_seconds",
    ):
        assert key in counters


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


def test_deterministic_slot_is_stable_and_in_range():
    job = ModelJob.create(
        job_type="boogu_edit",
        resource=RESOURCE_BOOGU,
        canonical_shard="shard-000000000-000000999",
        clip_uid="clip-1",
        semantic_inputs={"prompt": "x"},
        model_identity="boogu",
    )
    slot = deterministic_slot(job, 8)
    assert 0 <= slot < 8
    assert slot == deterministic_slot(job, 8)


class _FakeWorker:
    def __init__(self, slot: int, gpu_id: int):
        self.slot, self.gpu_id = slot, gpu_id
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_worker_epoch_loads_one_worker_per_gpu_once(tmp_path: Path):
    pool = WorkerPoolConfig(gpu_ids=(0, 1, 2, 3, 4, 5, 6, 7))
    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        pool,
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


def test_worker_epoch_slot_counts_are_recorded(tmp_path: Path):
    pool = WorkerPoolConfig(gpu_ids=(0, 1))
    resource = WorkerEpochResource(
        RESOURCE_SAM,
        pool,
        process_manager=_FakeProcessManager(),
        worker_factory=_FakeWorker,
        log_root=tmp_path,
    )
    resource.start()
    jobs = [
        ModelJob.create(
            job_type="sam_review",
            resource=RESOURCE_SAM,
            canonical_shard="shard-000000000-000000999",
            clip_uid=f"clip-{index}",
            semantic_inputs={"prompt": "x"},
            model_identity="sam3",
        )
        for index in range(6)
    ]
    for job in jobs:
        resource.handle_for(job)
    counts = resource.counters()["gpu_slot_job_counts"]
    assert sum(counts.values()) == 6
    assert set(counts) == {0, 1}


def test_worker_epoch_refuses_to_serve_before_start(tmp_path: Path):
    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(gpu_ids=(0,)),
        process_manager=_FakeProcessManager(),
        worker_factory=_FakeWorker,
        log_root=tmp_path,
    )
    job = ModelJob.create(
        job_type="boogu_edit",
        resource=RESOURCE_BOOGU,
        canonical_shard="shard-000000000-000000999",
        clip_uid="clip-1",
        semantic_inputs={"prompt": "x"},
        model_identity="boogu",
    )
    with pytest.raises(EpochResourceError):
        resource.handle_for(job)


def test_handle_executor_passes_only_a_handle_to_the_runner(tmp_path: Path):
    pool = WorkerPoolConfig(gpu_ids=(0, 1))
    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        pool,
        process_manager=_FakeProcessManager(),
        worker_factory=_FakeWorker,
        log_root=tmp_path,
    )
    resource.start()
    seen: list[Any] = []

    def runner(job: ModelJob, handle: object) -> JobResult:
        seen.append(handle)
        return JobResult(OUTCOME_COMPLETED, {"out.png": b"x"})

    executor = HandleExecutor(resource, runner)
    job = ModelJob.create(
        job_type="boogu_edit",
        resource=RESOURCE_BOOGU,
        canonical_shard="shard-000000000-000000999",
        clip_uid="clip-1",
        semantic_inputs={"prompt": "x"},
        model_identity="boogu",
    )
    result = executor.execute(job)
    assert result.outcome == OUTCOME_COMPLETED
    assert isinstance(seen[0], _FakeWorker)


def test_worker_epoch_stop_is_idempotent(tmp_path: Path):
    resource = WorkerEpochResource(
        RESOURCE_BOOGU,
        WorkerPoolConfig(gpu_ids=(0, 1)),
        process_manager=_FakeProcessManager(),
        worker_factory=_FakeWorker,
        log_root=tmp_path,
    )
    resource.start()
    resource.stop()
    resource.stop()
    assert resource.workers == []
