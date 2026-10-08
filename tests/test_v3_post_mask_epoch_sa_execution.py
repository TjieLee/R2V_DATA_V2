from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest

from r2v_data_v2.v3.post_mask_epoch_sa_execution import (
    SASamPipelineExecutor,
    StageAwareSASamExecutor,
)


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
            if job == "B":
                second.set()
            return job

        def persist_sam_job(self, job, inferred):
            if job == "A":
                writing.set()
                assert release.wait(5)
            return inferred

    executor = SASamPipelineExecutor(SlowWriter(), slot_count=1)
    try:
        executor.submit("A")
        assert writing.wait(5)
        executor.submit("B")
        assert second.wait(5)
        result = executor.collect()
        assert [item.job for item in result] == ["B"]
        release.set()
        assert executor.collect()[0].job == "A"
        diagnostics = executor.diagnostics()
        assert diagnostics["infer_persist_overlap_seconds"] > 0
        assert diagnostics["infer_slot_0_seconds"] > 0
    finally:
        release.set()
        executor.close()


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
            executor.submit(job)
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
            executor.submit(job)
        with pytest.raises(RuntimeError, match="capacity"):
            executor.submit(3)
        assert len(collect_all(executor, 3)) == 3
        executor.submit(3)
        assert executor.collect()[0].result == (3, 0)
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
    executor.submit("A")
    try:
        with pytest.raises(OSError, match="CPU failed"):
            executor.collect()
    finally:
        executor.close()
    assert executor.collect() == []


def test_close_drains_without_collect_and_joins_all_stages():
    writing, release, closed = Event(), Event(), Event()
    class Slow(Runner):
        def persist_sam_job(self, job, inferred):
            writing.set()
            assert release.wait(5)
            return inferred

    executor = SASamPipelineExecutor(Slow(), slot_count=1)
    for job in range(3):
        executor.submit(job)
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
        executor.submit(4)


def test_uncaught_inference_exception_reaches_main_thread():
    class Failure(Runner):
        def infer_sam_job(self, job, prepared, handle):
            if job == "A":
                raise ValueError("inference failed")
            return job
    executor = SASamPipelineExecutor(Failure(), slot_count=1)
    try:
        executor.submit("A")
        with pytest.raises(ValueError, match="inference failed"):
            executor.collect()
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
        executor.submit("attribute")
        assert executor.collect()[0].result == ("attribute", "backend")
        assert provider[0].sam_execution_diagnostics()["persist_count"] == 1
    finally:
        executor.close()


def test_interrupt_is_transported_and_close_drains_siblings():
    class Interrupted(Runner):
        def infer_sam_job(self, job, prepared, handle):
            if job == "A":
                raise KeyboardInterrupt()
            return job

    executor = SASamPipelineExecutor(Interrupted(), slot_count=1)
    executor.submit("A")
    executor.submit("B")
    try:
        with pytest.raises(KeyboardInterrupt):
            executor.collect()
    finally:
        executor.close()
    assert not any(thread.is_alive() for thread in executor._threads)
