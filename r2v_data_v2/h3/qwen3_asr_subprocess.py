from __future__ import annotations

import base64
import json
import os
import select
import subprocess
from pathlib import Path
from typing import TextIO

import numpy as np

from r2v_data_v2.h3.qwen3_asr import Qwen3ASRConfiguration


class PersistentQwen3ASRBackend:
    """Persistent JSONL adapter around the isolated qwen-asr environment."""

    def __init__(
        self,
        configuration: Qwen3ASRConfiguration,
        *,
        python_path: Path,
        worker_path: Path,
        timeout_seconds: float = 300.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Qwen3-ASR worker timeout must be positive")
        self.configuration = configuration
        self.python_path = python_path.expanduser().resolve(strict=True)
        self.worker_path = worker_path.expanduser().resolve(strict=True)
        if not self.python_path.is_file() or not self.worker_path.is_file():
            raise ValueError("Qwen3-ASR worker executable paths must be files")
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._request_index = 0

    def __enter__(self) -> PersistentQwen3ASRBackend:
        self._start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _read_response(self) -> dict[str, object]:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Qwen3-ASR worker is not running")
        ready, _, _ = select.select(
            [process.stdout],
            [],
            [],
            self.timeout_seconds,
        )
        if not ready:
            self.close(force=True)
            raise TimeoutError("Qwen3-ASR worker response timed out")
        line = process.stdout.readline()
        if not line:
            returncode = process.poll()
            raise RuntimeError(
                "Qwen3-ASR worker exited before producing a response"
                + ("" if returncode is None else f" (exit {returncode})")
            )
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError("Qwen3-ASR worker response must be an object")
        return payload

    def _start(self) -> None:
        if self._process is not None:
            return
        environment = os.environ.copy()
        # The isolated worker must not inherit the repository/SAM dependency overlay.
        environment.pop("PYTHONPATH", None)
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        command = [
            str(self.python_path),
            str(self.worker_path),
            "--model-path",
            self.configuration.local_model_path,
            "--device",
            self.configuration.device,
            "--dtype",
            self.configuration.dtype,
            "--max-inference-batch-size",
            str(self.configuration.max_inference_batch_size),
            "--max-new-tokens",
            str(self.configuration.max_new_tokens),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=environment,
        )
        self._process = process
        response = self._read_response()
        if response.get("request_id") != "startup" or response.get("status") != "ready":
            reason = str(response.get("reason") or "unknown startup failure")
            self.close(force=True)
            raise RuntimeError(f"Qwen3-ASR worker startup failed: {reason}")

    def transcribe(
        self,
        *,
        waveform: np.ndarray,
        sample_rate_hz: int,
    ) -> tuple[str, str | None]:
        if waveform.ndim != 1 or waveform.size == 0 or not np.isfinite(waveform).all():
            raise ValueError("Qwen3-ASR input must be finite, non-empty mono audio")
        if sample_rate_hz != 16000:
            raise ValueError("Qwen3-ASR worker requires 16 kHz model input")
        self._start()
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Qwen3-ASR worker is not running")
        self._request_index += 1
        request_id = f"transcribe-{self._request_index:08d}"
        audio = np.ascontiguousarray(waveform, dtype="<f4")
        request = {
            "request_id": request_id,
            "operation": "transcribe",
            "sample_rate_hz": sample_rate_hz,
            "sample_count": int(audio.size),
            "audio_f32le_base64": base64.b64encode(audio.tobytes()).decode("ascii"),
        }
        process.stdin.write(
            json.dumps(
                request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        process.stdin.flush()
        response = self._read_response()
        if response.get("request_id") != request_id:
            raise RuntimeError("Qwen3-ASR worker response request ID differs")
        if response.get("status") != "ok":
            raise RuntimeError(
                "Qwen3-ASR worker transcription failed: "
                + str(response.get("reason") or "unknown failure")
            )
        text = response.get("text")
        language = response.get("language")
        if not isinstance(text, str) or (
            language is not None and not isinstance(language, str)
        ):
            raise TypeError("Qwen3-ASR worker transcription payload is invalid")
        return text, language

    def close(self, *, force: bool = False) -> None:
        process = self._process
        if process is None:
            return
        self._process = None
        try:
            if not force and process.poll() is None and process.stdin is not None:
                process.stdin.write(
                    json.dumps(
                        {"request_id": "shutdown", "operation": "shutdown"},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                process.stdin.flush()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    force = True
            if force and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            for stream in (process.stdin, process.stdout):
                if isinstance(stream, TextIO):
                    stream.close()
                elif stream is not None:
                    stream.close()
