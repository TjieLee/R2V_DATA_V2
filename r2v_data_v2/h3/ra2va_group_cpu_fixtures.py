"""Explicit CPU fixtures: recorded model outputs, real media operations, no network."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver


class CPUFixtureFactory:
    def __init__(self, path, ffmpeg):
        self.path, self.ffmpeg = Path(path), ffmpeg

    @contextmanager
    def __call__(self, stage):
        data = json.loads(self.path.read_text())
        yield _FixtureBackend(data, self.ffmpeg)


class _FixtureBackend:
    def __init__(self, data, ffmpeg):
        self.data, self.ffmpeg = data, ffmpeg
        self.asr_index = {}
        self.config = MimoBackendConfig(api_key="cpu-fixture", transport="sglang",
            model="mimo-v2.6-flash-rl", max_completion_tokens=32768,
            media_resolver=MimoMediaResolver(mode="http", media_root=Path(data["media_root"]),
                                             media_base_url="http://cpu-fixture.invalid"))

    def materialize_full_audio(self, **kwargs):
        source = self.data["clips"][kwargs["clip_uid"]]["canonical_audio_path"]
        FFmpegAudioMediaBackend._publish_command([self.ffmpeg, "-nostdin", "-v", "error", "-i", source,
            "-map", "0:a:0", "-ar", "32000", "-ac", "2", "-c:a", "flac", "-sample_fmt", "s16", "-y"],
            kwargs["destination"])

    @staticmethod
    def _copy_pcm(source, destination):
        import soundfile as sf
        pcm, rate = sf.read(source, dtype="int16", always_2d=True)
        sf.write(destination, pcm, rate, subtype="PCM_16")

    def separate(self, **kwargs):
        for key in ("target_path", "residual_path"):
            self._copy_pcm(kwargs["source_audio_path"], kwargs[key])
        return SimpleNamespace(verification_state="unverified")

    def generate(self, source, raw):
        self._copy_pcm(source.source_audio_path, raw)
        return {"status": "ok", "cpu_fixture": True}

    def diarize(self, *, clip_uid, audio_path):
        return self.data["clips"][clip_uid]["diarization"]

    def bind_clip(self, clip_uid):
        self.clip_uid = clip_uid
        self.asr_index[clip_uid] = 0

    def transcribe(self, **kwargs):
        index = self.asr_index[self.clip_uid]
        self.asr_index[self.clip_uid] += 1
        row = self.data["clips"][self.clip_uid]["asr"][index]
        return row["text"], row["language"]

    def client_for(self, clip_uid):
        backend = self

        class Completions:
            def create(self, **payload):
                turn = {"MimoVisualDraft": "visual", "MimoJointAVAudioDraft": "joint"}[
                    payload["response_format"]["json_schema"]["name"]]
                return {"choices": [{"message": {"content": json.dumps(backend.data["clips"][clip_uid][turn])},
                                     "finish_reason": "stop"}], "usage": {}, "cpu_fixture": True}

        return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
