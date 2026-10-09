from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest

from r2v_data_v2.v3 import post_mask_epoch_sa_execution as sa_execution
from r2v_data_v2.v3.post_mask_epoch_resources import EpochResource
from r2v_data_v2.v3.post_mask_epoch_sa_execution import (
    SASamPipelineExecutor,
    StageAwareSASamExecutor,
)


def make_job(name):
    return SimpleNamespace(clip_uid=str(name), job_type="sam", attempt_index=1,
                           job_id=lambda: str(name))


class Runner:
    def prepare_sam_job(self, job):
        return job

    def infer_sam_job(self, job, prepared, handle):
        return (job, handle)

    def persist_sam_job(self, job, inferred):
        return inferred


def collect_all(executor, count):
    results = []
    while len(results) < count:
        results.extend(executor.collect())
    return results


def test_writer_does_not_hold_physical_slot_or_publish_early():
    writing, release, second = Event(), Event(), Event()

    class SlowWriter(Runner):
        def infer_sam_job(self, job, prepared, handle):
            if job.clip_uid == "B":
                second.set()
            return job

        def persist_sam_job(self, job, inferred):
            if job.clip_uid == "A":
                writing.set()
                assert release.wait(5)
            return inferred

    executor = SASamPipelineExecutor(SlowWriter(), slot_count=1)
    try:
        executor.submit(make_job("A"))
        assert writing.wait(5)
        executor.submit(make_job("B"))
        assert second.wait(5)
        result = executor.collect()
        assert [item.job.clip_uid for item in result] == ["B"]
        assert executor.diagnostics()["admitted"] == 1
        release.set()
        assert executor.collect()[0].job.clip_uid == "A"
        diagnostics = executor.diagnostics()
        assert diagnostics["infer_persist_overlap_seconds"] > 0
        assert diagnostics["infer_slot_0_seconds"] > 0
    finally:
        release.set()
        executor.close()


def test_writer_queue_wait_is_separate_from_actual_stage_wall(monkeypatch):
    writing, release, queued = Event(), Event(), Event()

    class SlowWriter(Runner):
        def persist_sam_job(self, job, inferred):
            if job.clip_uid == "A":
                writing.set()
                assert release.wait(5)
            return inferred

    executor = SASamPipelineExecutor(SlowWriter(), slot_count=1, persist_workers=1)
    put = executor._queues["persist"].put

    def record_put(item):
        put(item)
        if item is not None and item[0].clip_uid == "B":
            queued.set()

    monkeypatch.setattr(executor._queues["persist"], "put", record_put)
    try:
        executor.submit(make_job("A"))
        assert writing.wait(5)
        executor.submit(make_job("B"))
        assert queued.wait(5)
    finally:
        release.set()
        executor.close()
    assert len(collect_all(executor, 2)) == 2
    diagnostics = executor.diagnostics()
    assert diagnostics["persist_wait_seconds"] > 0
    assert diagnostics["infer_seconds"] > 0
    assert diagnostics["persist_seconds"] > 0
    assert diagnostics["persist_count"] == 2


def test_eight_slots_are_exclusive_and_all_live_workers_are_used():
    release, all_active = Event(), Event()
    lock = Lock()
    active, seen = set(), set()

    class Concurrent(Runner):
        def infer_sam_job(self, job, prepared, handle):
            with lock:
                assert handle not in active
                active.add(handle)
                seen.add(handle)
                if len(active) == 8:
                    all_active.set()
            assert release.wait(5)
            with lock:
                active.remove(handle)
            return job

    executor = SASamPipelineExecutor(Concurrent(), resource=SimpleNamespace(
        handle_for_slot=lambda slot: slot), slot_count=8)
    try:
        for job in range(24):
            executor.submit(make_job(job))
        assert all_active.wait(5)
        release.set()
        assert len(collect_all(executor, 24)) == 24
        assert seen == set(range(8))
        assert executor.diagnostics()["infer_active_peak"] == 8
    finally:
        release.set()
        executor.close()


def test_admission_includes_uncollected_results_and_bounds_stage_queues():
    executor = SASamPipelineExecutor(Runner(), slot_count=1, admission_capacity=3)
    try:
        for job in range(3):
            executor.submit(make_job(job))
        with pytest.raises(RuntimeError, match="capacity"):
            executor.submit(make_job(3))
        assert len(collect_all(executor, 3)) == 3
        fourth = make_job(3)
        executor.submit(fourth)
        assert executor.collect()[0].result == (fourth, 0)
        diagnostics = executor.diagnostics()
        assert diagnostics["admitted_peak"] == 3
        assert all(diagnostics[f"{stage}_queue_peak"] <= 3
                   for stage in ("prepare", "infer", "persist"))
    finally:
        executor.close()


@pytest.mark.parametrize("stage", ["prepare", "persist"])
def test_cpu_failure_reaches_collect_without_success(stage):
    class Failure(Runner):
        def prepare_sam_job(self, job):
            if stage == "prepare":
                raise OSError("CPU failed")
            return job

        def persist_sam_job(self, job, inferred):
            raise OSError("CPU failed")

    executor = SASamPipelineExecutor(Failure(), slot_count=1)
    executor.submit(make_job("A"))
    try:
        execution, = executor.collect()
        assert execution.result is None
        assert isinstance(execution.exception, OSError)
        assert str(execution.exception) == "CPU failed"
        assert executor.diagnostics()["admitted"] == 0
    finally:
        executor.close()
    assert executor.collect() == []


@pytest.mark.parametrize("stage", ["prepare", "infer", "persist"])
def test_ordinary_failure_does_not_lose_successful_sibling_in_same_wave(stage):
    class Failure(Runner):
        def fail(self, job, current_stage):
            if job.clip_uid == "A" and stage == current_stage:
                raise OSError("CPU persist failed")

        def prepare_sam_job(self, job):
            self.fail(job, "prepare")
            return job

        def infer_sam_job(self, job, prepared, handle):
            self.fail(job, "infer")
            return job

        def persist_sam_job(self, job, inferred):
            self.fail(job, "persist")
            return inferred

    executor = SASamPipelineExecutor(Failure(), slot_count=1)
    executor.submit(make_job("B"))
    executor.submit(make_job("A"))
    # Drain callbacks before collecting to guarantee both events share a wave.
    executor.close()
    failed, successful = executor.collect()
    assert failed.job.clip_uid == "A"
    assert failed.result is None
    assert isinstance(failed.exception, OSError)
    assert successful.job.clip_uid == "B"
    assert successful.result is successful.job
    assert successful.exception is None
    assert executor.diagnostics()["admitted"] == 0
    assert executor.collect() == []


def test_collect_sorts_only_the_already_completed_wave():
    executor = SASamPipelineExecutor(Runner(), slot_count=1,
                                     prepare_workers=1, persist_workers=1)
    for name in ("C", "B", "A"):
        executor.submit(make_job(name))
    executor.close()
    assert [execution.job.clip_uid for execution in executor.collect()] == ["A", "B", "C"]


def test_close_drains_without_collect_and_joins_all_stages():
    writing, release, closed = Event(), Event(), Event()
    class Slow(Runner):
        def persist_sam_job(self, job, inferred):
            writing.set()
            assert release.wait(5)
            return inferred

    executor = SASamPipelineExecutor(Slow(), slot_count=1)
    for job in range(3):
        executor.submit(make_job(job))
    assert writing.wait(5)
    closer = Thread(target=lambda: (executor.close(), closed.set()))
    closer.start()
    assert not closed.wait(0.05)
    release.set()
    closer.join(5)
    assert closed.is_set()
    assert not any(thread.is_alive() for thread in executor._threads)
    assert len(collect_all(executor, 3)) == 3
    assert executor.collect() == []
    executor.close()
    with pytest.raises(RuntimeError, match="closed"):
        executor.submit(make_job(4))


def test_uncaught_inference_exception_reaches_main_thread():
    class Failure(Runner):
        def infer_sam_job(self, job, prepared, handle):
            if job.clip_uid == "A":
                raise ValueError("inference failed")
            return job
    executor = SASamPipelineExecutor(Failure(), slot_count=1)
    try:
        executor.submit(make_job("A"))
        execution, = executor.collect()
        assert execution.result is None
        assert isinstance(execution.exception, ValueError)
        assert str(execution.exception) == "inference failed"
    finally:
        executor.close()


def test_stage_aware_wrapper_keeps_reference_edit_and_switches_only_when_drained():
    provider = [None]
    resource = SimpleNamespace(handle_for_slot=lambda slot: "backend")
    executor = StageAwareSASamExecutor(
        lambda job, handle: (job, handle), pipeline_runner=lambda: provider[0],
        slot_count=1, resource=resource,
    )
    reference = SimpleNamespace(clip_uid="reference", job_type="sam",
                                attempt_index=1, job_id=lambda: "reference")
    try:
        assert executor.capacity() == 1
        executor.submit(reference)
        provider[0] = Runner()
        with pytest.raises(RuntimeError, match="in.flight"):
            executor.capacity()
        assert executor.collect()[0].result == (reference, "backend")
        assert executor.capacity() == 3
        attribute = make_job("attribute")
        executor.submit(attribute)
        assert executor.collect()[0].result == (attribute, "backend")
        assert executor._outstanding == 0
        diagnostics = provider[0].sam_execution_diagnostics
        old_threads = executor._executor._threads[:]
        provider[0] = None
        assert executor.capacity() == 1
        assert not any(thread.is_alive() for thread in old_threads)
        assert diagnostics()["persist_count"] == 1
    finally:
        executor.close()


def test_interrupt_is_transported_and_close_drains_siblings():
    class Interrupted(Runner):
        def infer_sam_job(self, job, prepared, handle):
            if job.clip_uid == "A":
                raise KeyboardInterrupt()
            return job

    executor = SASamPipelineExecutor(Interrupted(), slot_count=1)
    executor.submit(make_job("A"))
    executor.submit(make_job("B"))
    try:
        with pytest.raises(KeyboardInterrupt):
            executor.collect()
    finally:
        executor.close()
    assert not any(thread.is_alive() for thread in executor._threads)


@pytest.mark.parametrize("fatal", [KeyboardInterrupt, SystemExit])
def test_stage_aware_fatal_collect_closes_threads_and_clears_capacity(fatal):
    class Interrupted(Runner):
        def persist_sam_job(self, job, inferred):
            if job.clip_uid == "A":
                raise fatal()
            return inferred

    runner = Interrupted()
    executor = StageAwareSASamExecutor(lambda job, handle: None,
                                       pipeline_runner=lambda: runner, slot_count=1)
    executor.submit(make_job("A"))
    executor.submit(make_job("B"))
    with pytest.raises(fatal):
        executor.collect()
    try:
        assert executor._outstanding == 0
        assert not any(thread.is_alive() for thread in executor._executor._threads)
        assert executor.collect() == []
        with pytest.raises(RuntimeError, match="closed"):
            executor.capacity()
    finally:
        executor.close()


def test_stage_aware_close_drains_uncollected_jobs_and_clears_outstanding():
    runner = Runner()
    executor = StageAwareSASamExecutor(lambda job, handle: None,
                                       pipeline_runner=lambda: runner, slot_count=1)
    executor.submit(make_job("A"))
    executor.submit(make_job("B"))
    executor.close()
    assert executor._outstanding == 0
    assert executor.collect() == []
    assert not any(thread.is_alive() for thread in executor._executor._threads)
    executor.close()


def test_stage_aware_ordinary_failure_drains_capacity_and_allows_stage_switch():
    class Failure(Runner):
        def persist_sam_job(self, job, inferred):
            if job.clip_uid == "A":
                raise OSError("persist failed")
            return inferred

    provider = [Failure()]
    executor = StageAwareSASamExecutor(lambda job, handle: job,
        pipeline_runner=lambda: provider[0], slot_count=1)
    try:
        executor.submit(make_job("A"))
        executor.submit(make_job("B"))
        executions = collect_all(executor, 2)
        assert sum(item.exception is not None for item in executions) == 1
        assert executor._outstanding == 0
        assert executor.diagnostics()["admitted"] == 0
        old_threads = executor._executor._threads[:]
        provider[0] = None
        assert executor.capacity() == 1
        assert not any(thread.is_alive() for thread in old_threads)
    finally:
        executor.close()


class ModeResource(EpochResource):
    def __init__(self, name, slots, events):
        super().__init__(name, process_manager=None)
        self.slot_count = slots
        self.events = events

    def _start(self):
        self.events.append((self.name, "start"))

    def _stop(self):
        if self.open:
            self.events.append((self.name, "stop"))

    def handle_for_slot(self, slot):
        assert self.open
        return (self.name, slot)


@pytest.mark.parametrize("workers_per_gpu", [2, 4])
def test_sa_mode_uses_only_child_pool_and_restores_reference_edit(workers_per_gpu):
    events, provider = [], [None]
    shared = ModeResource("reference", 2, events)
    child_pool = ModeResource("attribute", 2 * workers_per_gpu, events)
    resource = sa_execution.SASamModeResource(
        shared, lambda: child_pool, pipeline_runner=lambda: provider[0],
        workers_per_gpu=workers_per_gpu)
    resource.start()
    executor = StageAwareSASamExecutor(lambda job, handle: (job, handle),
        pipeline_runner=lambda: provider[0], slot_count=2, resource=resource)
    try:
        assert executor.capacity() == 2
        job = make_job("reference")
        executor.submit(job)
        assert executor.collect()[0].result == (job, ("reference", 0))
        provider[0] = Runner()
        assert executor.capacity() == 6 * workers_per_gpu
        assert events == [("reference", "start"), ("reference", "stop"),
                          ("attribute", "start")]
        assert not shared.open
        job = make_job("attribute")
        executor.submit(job)
        execution, = executor.collect()
        assert execution.result[1][0] == "attribute"
        diagnostics = executor.diagnostics()
        assert diagnostics["physical_slots"] == 2
        assert diagnostics["sam_processes"] == 2 * workers_per_gpu
        assert diagnostics["workers_per_gpu"] == workers_per_gpu
        provider[0] = None
        assert executor.capacity() == 2
        assert events[-2:] == [("attribute", "stop"), ("reference", "start")]
        assert not child_pool.open
    finally:
        executor.close()
        resource.stop()
    assert events[-1] == ("reference", "stop")


def test_cold_sa_mode_never_starts_parent_sam_handles():
    events, runner = [], Runner()
    shared = ModeResource("reference", 8, events)
    child_pool = ModeResource("attribute", 32, events)
    resource = sa_execution.SASamModeResource(shared, lambda: child_pool,
        pipeline_runner=lambda: runner, workers_per_gpu=4)
    resource.start()
    executor = StageAwareSASamExecutor(lambda *_: None,
        pipeline_runner=lambda: runner, slot_count=8, resource=resource)
    try:
        assert executor.capacity() == 96
        assert events == [("attribute", "start")]
        assert shared.start_count == 0
        executor.submit(make_job("attribute"))
        assert executor.collect()[0].exception is None
    finally:
        executor.close()
        resource.stop()
    assert events == [("attribute", "start"), ("attribute", "stop")]


@pytest.mark.parametrize("raw, expected", [("", 1), ("1", 1), ("2", 2), ("4", 4)])
def test_sa_workers_per_gpu_execution_only_option(monkeypatch, raw, expected):
    monkeypatch.setenv("POST_MASK_SA_SAM_WORKERS_PER_GPU", raw)
    assert sa_execution.resolve_sa_sam_workers_per_gpu() == expected


@pytest.mark.parametrize("raw", ["0", "3", "8", "bad"])
def test_sa_workers_per_gpu_rejects_unsupported_counts(monkeypatch, raw):
    monkeypatch.setenv("POST_MASK_SA_SAM_WORKERS_PER_GPU", raw)
    with pytest.raises(ValueError, match="1, 2, or 4"):
        sa_execution.resolve_sa_sam_workers_per_gpu()
