from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from r2v_data_v2.h3.ra2va_group_runtime import NativeBackendFactory


class SpeechModel:
    def __init__(self):
        self._process = SimpleNamespace(poll=lambda: None)
        self.loads, self.closes, self.batches, self.scalars = 0, 0, [], []

    def __enter__(self):
        self.loads += 1
        return self

    def close(self, **kwargs):
        self.closes += 1

    @staticmethod
    def result(waveform):
        return f"{int(waveform[0])}:{int(waveform[-1])}", "English"

    def transcribe_batch(self, *, waveforms, sample_rate_hz):
        assert sample_rate_hz == 16000
        self.batches.append([w.copy() for w in waveforms])
        return [self.result(w) for w in waveforms]

    def transcribe(self, *, waveform, sample_rate_hz):
        assert sample_rate_hz == 16000
        self.scalars.append(waveform.copy())
        return self.result(waveform)


@pytest.fixture
def speech_runtime(monkeypatch):
    model, decodes, configured_limits = SpeechModel(), [], []
    monkeypatch.delenv("QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE", raising=False)
    monkeypatch.delenv("QWEN3_ASR_DEVICE", raising=False)

    def isolated():
        import os
        configured_limits.append(os.environ["QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE"])
        return model

    def decode(path, start, end, *, ffmpeg):
        decodes.append((path, start, end, ffmpeg))
        return np.arange(round(end * 16000), dtype=np.float32), 16000

    monkeypatch.setattr("tools.run_h3_stem_qwen3_asr_shadow._isolated_backend", isolated)
    monkeypatch.setattr("r2v_data_v2.h3.qwen3_asr.load_qwen3_asr_model_input", decode)
    return model, decodes, configured_limits


def windows(count):
    return [{"source_start_sample": i * 320, "source_end_sample": i * 320 + 32,
             "source_sample_rate_hz": 32000} for i in range(count)]


def test_native_asr_batches_within_clip_decode_once_and_keeps_stage_model(speech_runtime, tmp_path):
    model, decodes, configured_limits = speech_runtime
    factory = NativeBackendFactory({"asr": {"environment": {}}}, mimo_config=None, ffmpeg="chosen-ffmpeg")
    with factory("asr") as backend:
        first = backend.transcribe_segments(audio_path=tmp_path / "a.wav", frame_count=32000, segments=windows(17))
        second = backend.transcribe_segments(audio_path=tmp_path / "b.wav", frame_count=64000, segments=windows(2))
        assert model.loads == 1 and model.closes == 0
    assert [len(batch) for batch in model.batches] == [8, 8, 1, 2]
    assert [row[0] for row in first] == [
        "0:15", "160:175", "320:335", "480:495", "640:655", "800:815",
        "960:975", "1120:1135", "1280:1295", "1440:1455", "1600:1615",
        "1760:1775", "1920:1935", "2080:2095", "2240:2255", "2400:2415", "2560:2575",
    ]
    assert second == [("0:15", "English"), ("160:175", "English")]
    assert decodes == [(tmp_path / "a.wav", 0, 1.0, "chosen-ffmpeg"),
                       (tmp_path / "b.wav", 0, 2.0, "chosen-ffmpeg")]
    assert configured_limits == ["8"] and not model.scalars
    assert model.closes == 1


def test_native_asr_scalar_option_still_decodes_stem_once(speech_runtime, tmp_path):
    model, decodes, limits = speech_runtime
    factory = NativeBackendFactory({"asr": {"environment": {}, "batch_size": 1}}, mimo_config=None)
    with factory("asr") as backend:
        result = backend.transcribe_segments(audio_path=tmp_path / "a.wav", frame_count=32000, segments=windows(3))
    assert result == [("0:15", "English"), ("160:175", "English"), ("320:335", "English")]
    assert len(decodes) == 1 and len(model.scalars) == 3 and not model.batches
    assert limits == ["1"]


@pytest.mark.parametrize("configuration,expected", [
    ({"environment": {"QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE": "4"}}, 4),
    ({"environment": {"QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE": "1"}}, 1),
    ({"batch_size": 8, "environment": {"QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE": "1"}}, 8),
    ({"batch_size": 2, "environment": {"QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE": "8"}}, 2),
])
def test_native_asr_config_precedence_matches_worker_environment(speech_runtime, tmp_path, configuration, expected):
    model, _, limits = speech_runtime
    original = json.loads(json.dumps(configuration))
    with NativeBackendFactory({"asr": configuration}, mimo_config=None)("asr") as backend:
        backend.transcribe_segments(audio_path=tmp_path / "a.wav", frame_count=32000, segments=windows(9))
    assert limits == [str(expected)]
    if expected == 1:
        assert len(model.scalars) == 9 and not model.batches
    else:
        assert [len(batch) for batch in model.batches] == [expected] * (9 // expected) + [9 % expected]
        assert not model.scalars
    assert configuration == original


def test_native_factory_defaults_duration_cap_without_changing_supplied_configuration():
    configuration = {"asr": {"environment": {}}}
    assert NativeBackendFactory(configuration, mimo_config=None).configuration["max_clip_duration_seconds"] == 20
    assert "max_clip_duration_seconds" not in configuration


def test_native_asr_empty_segments_need_no_audio_decode(speech_runtime, tmp_path):
    model, decodes, _ = speech_runtime
    with NativeBackendFactory({"asr": {"environment": {}}}, mimo_config=None)("asr") as backend:
        assert backend.transcribe_segments(audio_path=tmp_path / "absent.wav", frame_count=32000, segments=[]) == []
    assert not decodes and not model.batches and not model.scalars


def test_native_asr_rounds_frozen_32k_windows_like_t2va_and_clamps_eof(speech_runtime, tmp_path):
    model, _, _ = speech_runtime
    rows = [{"source_start_sample": 3, "source_end_sample": 11, "source_sample_rate_hz": 32000},
            {"source_start_sample": 31996, "source_end_sample": 32010, "source_sample_rate_hz": 32000}]
    with NativeBackendFactory({"asr": {"environment": {}}}, mimo_config=None)("asr") as backend:
        result = backend.transcribe_segments(audio_path=tmp_path / "a.wav", frame_count=32000, segments=rows)
    assert result == [("2:5", "English"), ("15998:15999", "English")]
    assert [len(w) for w in model.batches[0]] == [4, 2]
    assert all(w.flags.c_contiguous for w in model.batches[0])


@pytest.mark.parametrize("batch_size", [0, 9, 1.5, True])
def test_native_asr_rejects_invalid_batch_before_model_load(speech_runtime, batch_size):
    model, _, _ = speech_runtime
    factory = NativeBackendFactory({"asr": {"environment": {}, "batch_size": batch_size}}, mimo_config=None)
    with pytest.raises(ValueError, match="batch size"), factory("asr"):
        pass
    assert model.loads == 0


@pytest.mark.parametrize("start,end,rate", [(-2, 20, 32000), (40, 40, 32000),
                                          (32000, 32010, 32000), (0, 20, 16000)])
def test_native_asr_rejects_invalid_frozen_windows_without_inference(speech_runtime, tmp_path, start, end, rate):
    model, _, _ = speech_runtime
    with (NativeBackendFactory({"asr": {"environment": {}}}, mimo_config=None)("asr") as backend,
          pytest.raises(ValueError, match="(interval|sample rate)")):
        backend.transcribe_segments(audio_path=tmp_path / "a.wav", frame_count=32000,
            segments=[{"source_start_sample": start, "source_end_sample": end, "source_sample_rate_hz": rate}])
    assert not model.batches and not model.scalars


def test_native_asr_rejects_batch_result_count_mismatch(speech_runtime, tmp_path):
    model, _, _ = speech_runtime
    model.transcribe_batch = lambda **kwargs: []
    with (NativeBackendFactory({"asr": {"environment": {}}}, mimo_config=None)("asr") as backend,
          pytest.raises(ValueError, match="result count")):
        backend.transcribe_segments(audio_path=tmp_path / "a.wav", frame_count=32000, segments=windows(2))


@pytest.mark.parametrize("payload", [{"format": {}}, {"format": {"duration": "N/A"}},
                                     {"format": {"duration": "NaN"}}, {"format": {"duration": "0"}}])
def test_video_duration_probe_rejects_missing_or_invalid_format_duration(monkeypatch, tmp_path, payload):
    from r2v_data_v2.h3 import ra2va_group_pipeline as pipeline
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=json.dumps(payload)))
    with pytest.raises(ValueError, match="duration"):
        pipeline.probe_video_duration(tmp_path / "clip.mp4")


def test_real_ffprobe_short_video_duration_gate(tmp_path, monkeypatch):
    from r2v_data_v2.h3 import ra2va_group_pipeline as pipeline
    from tests.test_h3_ra2va_group_pipeline import ffmpeg, task_at

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        pytest.skip("real ffprobe is not installed")
    video = tmp_path / "short.mp4"
    subprocess.run([ffmpeg(), "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x32:r=10",
                    "-t", "0.3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(video)], check=True)
    task = task_at(tmp_path)
    task = type(task)(**{**vars(task), "target_video_path": str(video)})
    original, probes = subprocess.run, []

    def run(command, **kwargs):
        probes.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    factory = NativeBackendFactory({"max_clip_duration_seconds": 0.2}, mimo_config=None, ffprobe=ffprobe)
    with pipeline.GroupStageWorker("canonical", tmp_path / "out", backend_factory=factory, ffprobe=ffprobe) as worker:
        result = worker.process(task, {})
    assert result["status"] == "skipped" and result["reason"] == "clip_duration_over_0.2s"
    assert len(probes) == 1 and Path(probes[0][0]) == Path(ffprobe)
    assert not list((tmp_path / "out").rglob("full.flac"))
