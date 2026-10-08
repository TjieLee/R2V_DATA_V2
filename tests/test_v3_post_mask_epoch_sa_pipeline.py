from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from r2v_data_v2.v3.post_mask_epoch_sa_execution import SASamPipelineExecutor
from r2v_data_v2.v3.post_mask_epoch_subject_attributes import (
    SUBJECT_ATTRIBUTE_DISCOVERY_JOB,
    SUBJECT_ATTRIBUTE_SAM_PROBE_JOB,
)
from tests import test_v3_post_mask_epoch_subject_attributes as fixture


def prepared_runner(tmp_path, monkeypatch, attributes):
    config, storage = fixture._storage_variant(tmp_path, monkeypatch, "run-pipeline")
    runner = fixture._runner(config, storage, tmp_path)
    qwen = fixture._QwenClient(discoveries=[fixture._human_discovery(attributes=attributes)])
    fixture._scheduler(runner, fixture._SerialQwenExecutor(runner, qwen),
                       fixture._swallow(SUBJECT_ATTRIBUTE_DISCOVERY_JOB, runner)).run(
                           runner.seed_jobs())
    jobs = runner.seed_jobs()
    assert all(job.job_type == SUBJECT_ATTRIBUTE_SAM_PROBE_JOB for job in jobs)
    return config, storage, runner, jobs


def test_real_runner_masks_are_readonly_independent_before_slot_reuse(tmp_path, monkeypatch):
    config, storage, runner, jobs = prepared_runner(tmp_path, monkeypatch, (fixture.ACCESSORY,))
    masks = fixture._usable_sam(storage, slot=0)
    backend = fixture._SamBackend(masks)
    prepared = runner.prepare_sam_job(jobs[0])
    inferred = runner.infer_sam_job(jobs[0], prepared, backend)
    masks[0][:] = False
    result = runner.persist_sam_job(jobs[0], inferred)
    assert result.payload["status"] == "sam"
    stored = runner._load_sam_masks(result.payload)
    assert np.any(stored[0])
    assert not inferred["masks"][0].flags.writeable
    assert runner._committed_payload_or_none(jobs[0]) is None
    # Successful NPY publication without a receipt is still a crash window,
    # not a completed job; cold resume must execute the same model identity.
    fresh = fixture._runner(config, storage, tmp_path)
    resumed = next(job for job in fresh.seed_jobs() if job.job_id() == jobs[0].job_id())
    resumed_backend = fixture._SamBackend(fixture._usable_sam(storage, slot=0))
    assert fresh.run(resumed, resumed_backend).payload["status"] == "sam"
    assert resumed_backend.calls == 1


def test_real_runner_writer_overlap_and_success_receipt_cold_resume(tmp_path, monkeypatch):
    config, storage, runner, jobs = prepared_runner(
        tmp_path, monkeypatch, (fixture.ACCESSORY, fixture.SCARF))
    writing, release, second = Event(), Event(), Event()
    publish = runner._publish_masks
    first_id = jobs[0].job_id()
    def slow_publish(job_id, masks):
        if job_id == first_id:
            writing.set()
            assert release.wait(5)
        return publish(job_id, masks)
    monkeypatch.setattr(runner, "_publish_masks", slow_publish)
    class Backend(fixture._SamBackend):
        def segment_frame(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                second.set()
            return fixture._usable_sam(storage, slot=0)
    backend = Backend()
    executor = SASamPipelineExecutor(runner, slot_count=1,
        resource=SimpleNamespace(handle_for_slot=lambda slot: backend))
    try:
        executor.submit(jobs[0])
        assert writing.wait(5)
        executor.submit(jobs[1])
        assert second.wait(5)
        assert runner._committed_payload_or_none(jobs[0]) is None
        release.set()
        outcomes = []
        while len(outcomes) < 2:
            outcomes.extend(executor.collect())
    finally:
        release.set()
        executor.close()
    # Feed persisted outcomes through the real ledger/scheduler receipt path.
    class ReadyExecutor:
        def execute_batch(self, batch):
            return {item.job.job_id(): item for item in outcomes}
    fixture._scheduler(runner, fixture._SerialQwenExecutor(runner, fixture._QwenClient()),
                       fixture._swallow(SUBJECT_ATTRIBUTE_SAM_PROBE_JOB, runner),
                       ReadyExecutor()).run(jobs)
    for job in jobs:
        assert runner._committed_payload_or_none(job)["status"] == "sam"
    fresh = fixture._runner(config, storage, tmp_path)
    assert not any(job.job_id() in {old.job_id() for old in jobs}
                   for job in fresh.seed_jobs())


def test_writer_failure_no_receipt_and_cold_resume_retries_same_job(tmp_path, monkeypatch):
    config, storage, runner, jobs = prepared_runner(tmp_path, monkeypatch, (fixture.ACCESSORY,))
    def failure(*args):
        raise OSError("persist EIO")
    monkeypatch.setattr(runner, "_publish_masks", failure)
    executor = SASamPipelineExecutor(runner, slot_count=1,
        resource=SimpleNamespace(handle_for_slot=lambda slot: fixture._SamBackend(
            fixture._usable_sam(storage, slot=0))))
    try:
        with pytest.raises(OSError, match="persist EIO"):
            fixture._scheduler(runner, fixture._SerialQwenExecutor(runner, fixture._QwenClient()),
                               sam=executor).run(jobs)
    finally:
        executor.close()
    assert runner._committed_payload_or_none(jobs[0]) is None
    fresh = fixture._runner(config, storage, tmp_path)
    assert jobs[0].job_id() in {job.job_id() for job in fresh.seed_jobs()}
