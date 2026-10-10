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


def test_asr_uses_native_segments_interface_without_rebuilding_metadata(tmp_path):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    fake = PCMBackends()
    rows = [{"segment_id": "s1", "start_time": 0.1, "end_time": 0.3,
             "source_start_sample": 3200, "source_end_sample": 9600,
             "source_sample_rate_hz": 32000, "source_speaker_cluster_id": "speaker_0"},
            {"segment_id": "s2", "start_time": 0.5, "end_time": 0.7,
             "source_start_sample": 16000, "source_end_sample": 22400,
             "source_sample_rate_hz": 32000, "source_speaker_cluster_id": "speaker_1"}]
    rows = [{**row, "identity_scope": "unresolved", "direct_anchor_seconds": 0,
             "cluster_binding_status": "unbound", "overlapping_visible_entities": [],
             "direct_support_seconds_by_entity": {}, "competing_visible_speaker_evidence": []}
            for row in rows]
    speech = {"path": str(tmp_path / "not-decoded-by-pipeline.wav"), "frame_count": 32000}
    calls = []

    def transcribe_segments(**kwargs):
        calls.append(kwargs)
        return [("  Exact first.  ", "English"), (" ", "Chinese")]

    fake.transcribe_segments = transcribe_segments
    fake.transcribe = lambda **kw: pytest.fail("native interface must not use scalar fallback")
    with GroupStageWorker("asr", tmp_path / "out", backend_factory=fake.factory) as worker:
        result = worker.process(task_at(tmp_path), {"diarizen": {"segments": rows},
            "resolve": {"speech": speech}, "canonical": speech})
    assert calls == [{"audio_path": tmp_path / "not-decoded-by-pipeline.wav",
                      "frame_count": 32000, "segments": rows}]
    segments = result["job"]["segments"]
    assert [s["segment_id"] for s in segments] == ["s1", "s2"]
    assert [(s["source_start_sample"], s["source_end_sample"]) for s in segments] == [(3200, 9600), (16000, 22400)]
    assert [s["asr_status"] for s in segments] == ["transcribed", "empty"]
    assert [s["asr_text"] for s in segments] == ["  Exact first.  ", None]
    assert [s["asr_language"] for s in segments] == ["English", None]


def test_asr_rejects_native_result_count_mismatch(tmp_path):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    fake = PCMBackends()
    fake.transcribe_segments = lambda **kw: []
    speech = {"path": str(tmp_path / "absent.wav"), "frame_count": 32000}
    with (GroupStageWorker("asr", tmp_path, backend_factory=fake.factory) as worker,
          pytest.raises(ValueError, match="result count")):
        worker.process(task_at(tmp_path), {"diarizen": {"segments": [{"segment_id": "s1"}]},
            "resolve": {"speech": speech}, "canonical": speech})


@pytest.mark.parametrize("configured_cap,duration,reason", [
    (None, 20.000001, "clip_duration_over_20s"),
    (2.0, 2.000001, "clip_duration_over_2s"),
])
def test_canonical_skips_overlong_video_before_audio_and_reuses_receipt(
    tmp_path, monkeypatch, configured_cap, duration, reason
):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    fake = PCMBackends()
    fake.materialize_full_audio = lambda **kw: pytest.fail("overlong clip must not extract audio")

    class Factory:
        configuration = {} if configured_cap is None else {"max_clip_duration_seconds": configured_cap}

        def __call__(self, stage):
            assert stage == "canonical"
            return fake.factory(stage)

    probes = []

    def probe(path, *, ffprobe):
        probes.append((path, ffprobe))
        return duration

    monkeypatch.setattr("r2v_data_v2.h3.ra2va_group_pipeline.probe_video_duration", probe)
    task = task_at(tmp_path)
    with GroupStageWorker("canonical", tmp_path / "out", backend_factory=Factory(),
                          ffprobe="selected-ffprobe", duration_probe=probe if configured_cap is None else None) as worker:
        result = worker.process(task, {})
        assert result["status"] == "skipped" and result["reason"] == reason
        assert worker.process(task, {}) == result
    assert probes == [(type(tmp_path)(task.target_video_path), "selected-ffprobe")]
    assert not list((tmp_path / "out").rglob("full.flac"))


def test_canonical_duration_equal_to_default_limit_is_admitted(tmp_path):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    fake = PCMBackends()
    with GroupStageWorker("canonical", tmp_path, backend_factory=fake.factory,
                          duration_probe=lambda *args, **kw: 20.0) as worker:
        result = worker.process(task_at(tmp_path), {})
    assert result["frame_count"] == 32000


def test_legacy_fake_factory_does_not_probe_missing_video(tmp_path, monkeypatch):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    monkeypatch.setattr("r2v_data_v2.h3.ra2va_group_pipeline.probe_video_duration",
                        lambda *args, **kwargs: pytest.fail("legacy fake must not probe"), raising=False)
    fake = PCMBackends()
    with GroupStageWorker("canonical", tmp_path, backend_factory=fake.factory) as worker:
        result = worker.process(task_at(tmp_path), {})
    assert result["frame_count"] == 32000


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf"), "N/A", None])
def test_canonical_invalid_duration_fails_before_audio(tmp_path, duration):
    from r2v_data_v2.h3.ra2va_group_pipeline import GroupStageWorker
    fake = PCMBackends()
    fake.materialize_full_audio = lambda **kw: pytest.fail("invalid duration must not extract audio")
    with (GroupStageWorker("canonical", tmp_path, backend_factory=fake.factory,
                           duration_probe=lambda *args, **kw: duration) as worker,
          pytest.raises(ValueError, match="duration")):
        worker.process(task_at(tmp_path), {})
