from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from r2v_data_v2.h3.ra2va_group_source import adapt_production_sample
from r2v_data_v2.v3.production_export import ProductionSample
from tests.test_h3_ra2va_group_source import sample


def task_at(root, uid="clip-a"):
    values = sample(uid)
    return adapt_production_sample(ProductionSample.model_validate(values), root, 0)


class PCMBackends:
    def __init__(self):
        self.opened, self.closed, self.calls = [], [], []

    @contextmanager
    def factory(self, stage):
        self.opened.append(stage)
        yield self
        self.closed.append(stage)

    def materialize_full_audio(self, **kw):
        sf.write(kw["destination"], np.arange(64000, dtype=np.int16).reshape(-1, 2), 32000,
                 subtype="PCM_16")

    def separate(self, **kw):
        self.calls.append(kw)
        pcm, rate = sf.read(kw["source_audio_path"], dtype="int16", always_2d=True)
        sf.write(kw["target_path"], pcm, rate, subtype="PCM_16")
        sf.write(kw["residual_path"], pcm, rate, subtype="PCM_16")
        return SimpleNamespace(verification_state="success")

    def generate(self, source, raw):
        self.calls.append(source)
        pcm, rate = sf.read(source.source_audio_path, dtype="int16", always_2d=True)
        sf.write(raw, pcm, rate, subtype="PCM_16")
        return {"status": "ok", "model_runtime_seconds": 0.1}

    def diarize(self, **kw):
        return [{"start_time": 0.1, "end_time": 0.3, "speaker_label": "voice-b"},
                {"start_time": 0.5, "end_time": 0.7, "speaker_label": "voice-a"}]

    def transcribe(self, **kw):
        self.calls.append(kw)
        return "Exact words.", "English"


def test_native_job_preserves_v3_contract_without_input_hashes(tmp_path):
    from r2v_data_v2.h3.ra2va_group_pipeline import frozen_job
    task = task_at(tmp_path)
    job = frozen_job(task, {"path": str(tmp_path / "full.flac"), "frame_count": 32000}, [])
    data = job.model_dump(mode="json")
    assert job.reference_subjects[0].entity_id == "e7"
    assert job.reference_subjects[1].owner_entity_id == "e7"
    assert [r.source_image_id for r in job.reference_images] == ["image_1", "image_2", "image_3"]
    assert [r.picture_label for r in job.reference_images] == ["<Picture 1>", "<Picture 2>", "<Picture 3>"]
    assert not any(token in json.dumps(data) for token in ("sha256", "fingerprint", "seed"))
    assert job.binding_evidence_mode == "none"
    assert job.target_duration_seconds == 1


def test_audio_stages_reuse_context_and_preserve_exact_sample_windows(tmp_path):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    fake = PCMBackends()
    task = task_at(tmp_path)
    payloads = {}
    for stage in ("canonical", "sam", "auk", "resolve", "diarizen", "asr"):
        with GroupStageWorker(stage, tmp_path / "artifacts", backend_factory=fake.factory,
                              ffmpeg=ffmpeg()) as worker:
            payloads[stage] = worker.process(task, payloads)
            if stage == "canonical":
                worker.process(task_at(tmp_path, "clip-b"), {})
    assert fake.opened.count("canonical") == fake.closed.count("canonical") == 1
    assert len(fake.calls[0:2]) == 2
    assert fake.calls[0]["prompt"] == "music soundtrack"
    assert fake.calls[1]["prompt"] == "human voices"
    assert fake.calls[1]["source_audio_path"] == fake.calls[0]["residual_path"]
    resolved = payloads["resolve"]
    assert resolved["speech"]["path"] == payloads["auk"]["speech"]["path"]
    assert resolved["music"] == payloads["sam"]["music"]
    assert resolved["sfx"] == payloads["sam"]["sfx"]
    job = payloads["asr"]["job"]
    assert [s["source_speaker_cluster_id"] for s in job["segments"]] == ["speaker_0", "speaker_1"]
    assert [(s["source_start_sample"], s["source_end_sample"]) for s in job["segments"]] == [(3200, 9600), (16000, 22400)]
    assert all(s["current_entity_id"] is None for s in job["segments"])
    assert all(s["asr_text"] == "Exact words." for s in job["segments"])
    assert all(s["source_sample_rate_hz"] == 32000 for s in job["segments"])


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def test_empty_diarization_does_not_invent_asr_or_speaker(tmp_path):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    fake = PCMBackends()
    fake.diarize = lambda **kw: []
    task = task_at(tmp_path)
    payloads = {}
    for stage in ("canonical", "sam", "auk", "resolve", "diarizen", "asr"):
        with GroupStageWorker(stage, tmp_path / "out", backend_factory=fake.factory, ffmpeg=ffmpeg()) as worker:
            payloads[stage] = worker.process(task, payloads)
    assert payloads["asr"]["job"]["segments"] == []
    assert not any(isinstance(call, dict) and "waveform" in call for call in fake.calls)


def test_no_missing_stage_is_silently_substituted(tmp_path):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    with (GroupStageWorker("resolve", tmp_path, backend_factory=PCMBackends().factory) as worker,
          pytest.raises(KeyError)):
        worker.process(task_at(tmp_path), {})


def test_sam_unverified_is_preserved_not_promoted_to_success(tmp_path):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    fake = PCMBackends()
    original = fake.separate

    def unverified(**kwargs):
        original(**kwargs)
        return SimpleNamespace(verification_state="unverified")

    fake.separate = unverified
    with GroupStageWorker("canonical", tmp_path, backend_factory=fake.factory) as worker:
        canonical = worker.process(task_at(tmp_path), {})
    with GroupStageWorker("sam", tmp_path, backend_factory=fake.factory, ffmpeg=ffmpeg()) as worker:
        result = worker.process(task_at(tmp_path), {"canonical": canonical})
    assert result["separation_state"] == "unverified"
    assert result["model_call_count"] == 2


def test_audio_stage_recovery_does_not_overwrite_partial_or_repeat_completed_work(tmp_path):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    fake = PCMBackends()
    original = fake.materialize_full_audio
    paths = []

    def interrupted(**kwargs):
        assert not kwargs["destination"].exists()
        original(**kwargs)
        paths.append(kwargs["destination"])
        raise SystemExit("crash before stage receipt")

    fake.materialize_full_audio = interrupted
    task = task_at(tmp_path)
    with (GroupStageWorker("canonical", tmp_path / "out", backend_factory=fake.factory) as worker,
          pytest.raises(SystemExit)):
        worker.process(task, {})

    def recovered(**kwargs):
        assert not kwargs["destination"].exists()
        paths.append(kwargs["destination"])
        original(**kwargs)

    fake.materialize_full_audio = recovered
    with GroupStageWorker("canonical", tmp_path / "out", backend_factory=fake.factory) as worker:
        result = worker.process(task, {})
    assert len(paths) == 2 and paths[0] != paths[1]
    fake.materialize_full_audio = lambda **kwargs: pytest.fail("saved stage must not execute again")
    with GroupStageWorker("canonical", tmp_path / "out", backend_factory=fake.factory) as worker:
        assert worker.process(task, {}) == result
