"""Persistent existing model backends for native Group stage inputs.

This is adapter wiring, not GPU scheduling or a model-server launcher.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.auk_speech_shadow import AukConfiguration, PersistentAukBackend


class GroupAukBackend(PersistentAukBackend):
    def generate(self, job, raw_path):
        self._send({"request_id": job.clip_uid, "operation": "generate_group",
            "raw_output_path": str(raw_path), "source_audio_path": job.source_audio_path,
            "source_duration_seconds": job.source_duration_seconds})
        response = self._read()
        if response.get("request_id") != job.clip_uid or response.get("status") not in {"ok", "failed"}:
            raise ValueError("AuK worker response identity/status differs")
        return response


class _SpeechBackend:
    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        from r2v_data_v2.h3.t2va_full_speech import _inference
        method = getattr(self.backend, name)
        return lambda **kwargs: _inference(self.backend, lambda: method(**kwargs))


class NativeBackendFactory:
    """Use supplied runtime configurations, never rebuild hash-heavy inventories."""

    def __init__(self, configuration, *, mimo_config, ffmpeg="ffmpeg", ffprobe="ffprobe"):
        self.configuration, self.mimo_config = configuration, mimo_config
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe

    @contextmanager
    def __call__(self, stage):
        if stage == "canonical":
            yield FFmpegAudioMediaBackend(ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)
        elif stage == "sam":
            from r2v_data_v2.h3.sam_audio_stem_shadow import (
                OfficialSAMAudioBackend,
                SAMAudioModelConfiguration,
            )
            backend = OfficialSAMAudioBackend(SAMAudioModelConfiguration.model_validate(self.configuration["sam"]))
            try:
                backend._load()
                yield backend
            finally:
                backend._model = backend._processor = None
                if backend._torch is not None:
                    backend._torch.cuda.empty_cache()
        elif stage == "auk":
            with GroupAukBackend(AukConfiguration.model_validate(self.configuration["auk"])) as backend:
                yield backend
        elif stage in {"diarizen", "asr"}:
            from r2v_data_v2.h3.t2va_full_speech import asr_worker, diarizen_worker
            factory = diarizen_worker if stage == "diarizen" else asr_worker
            with factory(self.configuration[stage]) as worker:
                yield _SpeechBackend(worker.backend)
        elif stage == "mimo":
            from openai import OpenAI
            with OpenAI(api_key=self.mimo_config.api_key, base_url=self.mimo_config.base_url,
                        timeout=self.mimo_config.timeout_seconds, max_retries=0) as client:
                yield SimpleNamespace(config=self.mimo_config, client=client)
        else:
            raise ValueError(f"no native backend for {stage}")
