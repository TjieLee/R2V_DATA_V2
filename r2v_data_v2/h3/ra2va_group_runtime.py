"""Persistent existing model backends for native Group stage inputs.

This is adapter wiring, not GPU scheduling or a model-server launcher.
"""

from __future__ import annotations

import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.auk_speech_shadow import (
    FIXED_INSTRUCTION,
    PersistentAukBackend,
    worker_path,
)


class RuntimeSettings(SimpleNamespace):
    """Execution settings only, without legacy inventory/provenance fields."""
    def model_dump(self, **kwargs):
        return vars(self)


def _start_jsonl(backend, script):
    environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    environment.pop("PYTHONPATH", None)
    backend.process = subprocess.Popen([backend.configuration.python_path, str(script)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True, bufsize=1, env=environment)
    try:
        backend._send(backend.configuration.model_dump())
        result = backend._read()
        if result.get("status") != "ready":
            raise RuntimeError(f"native backend startup failed: {result}")
    except BaseException:
        backend.close()
        raise
    return backend


class GroupAukBackend(PersistentAukBackend):
    def __enter__(self):
        return _start_jsonl(self, worker_path())

    def _read(self):
        try:
            return super()._read()
        except (TimeoutError, BrokenPipeError, EOFError) as error:
            raise RuntimeError(f"Native model process interrupted: {error}") from error

    def generate(self, job, raw_path):
        self._send({"request_id": job.clip_uid, "operation": "generate_group",
            "raw_output_path": str(raw_path), "source_audio_path": job.source_audio_path,
            "source_duration_seconds": job.source_duration_seconds})
        response = self._read()
        if response.get("request_id") != job.clip_uid or response.get("status") not in {"ok", "failed"}:
            raise ValueError("AuK worker response identity/status differs")
        if "out of memory" in response.get("reason", "").lower():
            raise RuntimeError(f"AuK inference process failed: {response['reason']}")
        return response


class GroupDiariZenBackend(GroupAukBackend):
    def __enter__(self):
        return _start_jsonl(self, Path(__file__).resolve().parents[2] / "tools/run_h3_ra2va_group_diarizen_worker.py")

    def diarize(self, *, clip_uid, audio_path):
        self._send({"request_id": clip_uid, "audio_path": str(audio_path)})
        response = self._read()
        if response.get("request_id") != clip_uid:
            raise RuntimeError("DiariZen native response belongs to another task")
        if response["status"] != "ready":
            if "out of memory" in response["reason"].lower():
                raise RuntimeError(response["reason"])
            raise ValueError(response["reason"])
        return response["segments"]


class _SpeechBackend:
    def __init__(self, backend, *, ffmpeg, batch_size):
        self.backend = backend
        self.ffmpeg, self.batch_size = ffmpeg, batch_size

    def __getattr__(self, name):
        from r2v_data_v2.h3.t2va_full_speech import _inference
        method = getattr(self.backend, name)
        return lambda **kwargs: _inference(self.backend, lambda: method(**kwargs))

    def transcribe_segments(self, *, audio_path, frame_count, segments):
        import numpy as np

        from r2v_data_v2.h3.qwen3_asr import load_qwen3_asr_model_input

        if not segments:
            return []
        waveform, rate = load_qwen3_asr_model_input(
            Path(audio_path), 0, frame_count / 32000, ffmpeg=self.ffmpeg,
        )
        windows = []
        for segment in segments:
            if segment["source_sample_rate_hz"] != 32000:
                raise ValueError("Qwen3 ASR source sample rate must be 32000")
            # Match T2VA's rounding after mapping the frozen canonical window.
            start = round(segment["source_start_sample"] / 32000 * 16000)
            end = min(round(segment["source_end_sample"] / 32000 * 16000), len(waveform))
            if segment["source_start_sample"] < 0 or end <= start:
                raise ValueError("Qwen3 ASR source interval has no decoded samples")
            windows.append(np.ascontiguousarray(waveform[start:end]))
        results = []
        for offset in range(0, len(windows), self.batch_size):
            batch = windows[offset:offset + self.batch_size]
            transcribed = ([self.transcribe(waveform=batch[0], sample_rate_hz=rate)]
                           if self.batch_size == 1 else
                           self.transcribe_batch(waveforms=batch, sample_rate_hz=rate))
            if len(transcribed) != len(batch):
                raise ValueError("Qwen3 ASR batch result count differs")
            results.extend(transcribed)
        return results


class NativeBackendFactory:
    """Use supplied runtime configurations, never rebuild hash-heavy inventories."""

    def __init__(self, configuration, *, mimo_config, ffmpeg="ffmpeg", ffprobe="ffprobe"):
        self.configuration = {"max_clip_duration_seconds": 20, **configuration}
        self.mimo_config = mimo_config
        self.ffmpeg, self.ffprobe = ffmpeg, ffprobe

    @contextmanager
    def __call__(self, stage):
        if stage == "canonical":
            yield FFmpegAudioMediaBackend(ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)
        elif stage == "sam":
            from r2v_data_v2.h3.sam_audio_stem_shadow import OfficialSAMAudioBackend
            values = self.configuration["sam"]
            backend = OfficialSAMAudioBackend(RuntimeSettings(**values,
                model_config_path=str(Path(values["model_path"]) / "config.json"), device="cuda:0",
                predict_spans=False, span_predictor_path=None, reranking_candidates=1))
            try:
                backend._load()
                yield backend
            finally:
                backend._model = backend._processor = None
                if backend._torch is not None:
                    backend._torch.cuda.empty_cache()
        elif stage == "auk":
            values = self.configuration["auk"]
            settings = RuntimeSettings(**values, device="cuda:0", dtype="bf16", instruction=FIXED_INSTRUCTION,
                config_path=str(Path(values["checkpoint_path"]).with_name("config.yaml")),
                nfe=32, cfg_strength=2.0, sway_sampling_coef=-1.0, seed=0)
            with GroupAukBackend(settings) as backend:
                yield backend
        elif stage == "diarizen":
            with GroupDiariZenBackend(RuntimeSettings(**self.configuration[stage])) as backend:
                yield backend
        elif stage == "asr":
            from r2v_data_v2.h3.t2va_full_speech import asr_worker
            values = self.configuration[stage]
            environment = dict(values.get("environment", {}))
            try:
                batch_size = (values["batch_size"] if "batch_size" in values else
                              int(environment.get("QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE", 8)))
            except (TypeError, ValueError) as error:
                raise ValueError("Qwen3 ASR batch size must be an integer from 1 to 8") from error
            if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 8:
                raise ValueError("Qwen3 ASR batch size must be an integer from 1 to 8")
            environment["QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE"] = str(batch_size)
            with asr_worker({**values, "ffmpeg": self.ffmpeg, "environment": environment}) as worker:
                yield _SpeechBackend(worker.backend, ffmpeg=self.ffmpeg, batch_size=batch_size)
        elif stage == "mimo":
            from openai import OpenAI
            with OpenAI(api_key=self.mimo_config.api_key, base_url=self.mimo_config.base_url,
                        timeout=self.mimo_config.timeout_seconds, max_retries=0) as client:
                yield SimpleNamespace(config=self.mimo_config, client=client)
        else:
            raise ValueError(f"no native backend for {stage}")


def load_native_configuration(path, endpoint_index=0):
    from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
    values = json.loads(Path(path).read_text())
    values["log_root"] = os.path.expandvars(values["log_root"])
    mimo = values["mimo"]
    config = MimoBackendConfig(api_key=os.environ.get("MIMO_API_KEY", "local"), transport="sglang",
        model="mimo-v2.6-flash-rl", base_url=f"http://127.0.0.1:{mimo['ports'][endpoint_index]}/v1",
        temperature=0, thinking="disabled", max_completion_tokens=32768,
        timeout_seconds=mimo.get("timeout_seconds", 900),
        media_resolver=MimoMediaResolver(mode="http", media_root=Path(mimo["media_root"]),
                                         media_base_url=mimo["media_base_url"]))
    return values, NativeBackendFactory(values, mimo_config=config,
        ffmpeg=values.get("ffmpeg", "ffmpeg"), ffprobe=values.get("ffprobe", "ffprobe"))
