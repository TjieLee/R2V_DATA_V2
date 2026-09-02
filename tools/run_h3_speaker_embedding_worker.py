#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from r2v_data_v2.h3.audio_backends import fingerprint_local_model_path

_ANALYSIS_RESAMPLE_POLICY = (
    "h3_speaker_ecapa_torchaudio_kaiser_32k_stereo_to_16k_mono_v1"
)
_NATIVE_INPUT_POLICY = "h3_speaker_ecapa_native_16k_mono_passthrough_v1"


def _prepare_model_input(
    waveform: Any,
    sample_rate: int,
    *,
    torch: Any,
    torchaudio: Any,
) -> tuple[Any, str]:
    if waveform.ndim != 2 or waveform.shape[1] <= 0:
        raise ValueError("speaker input waveform must be channel-first audio")
    if not torch.isfinite(waveform).all().item():
        raise ValueError("speaker input waveform must be finite and non-empty")
    if sample_rate == 32000 and waveform.shape[0] == 2:
        mono = waveform.mean(dim=0, keepdim=True)
        return (
            torchaudio.functional.resample(
                mono,
                orig_freq=32000,
                new_freq=16000,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            ),
            _ANALYSIS_RESAMPLE_POLICY,
        )
    if sample_rate == 16000 and waveform.shape[0] == 1:
        return waveform, _NATIVE_INPUT_POLICY
    raise ValueError("speaker input must be canonical 32 kHz stereo or 16 kHz mono")


def validate_speaker_model_path(model_path: Path) -> Path:
    source = model_path.expanduser().resolve(strict=True)
    if not source.is_dir() or not (source / "hyperparams.yaml").is_file():
        raise FileNotFoundError(
            "explicit local SpeechBrain model must contain hyperparams.yaml"
        )
    return source


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent local SpeechBrain speaker embedding worker"
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-identifier", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


class SpeechBrainWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        model_path = validate_speaker_model_path(args.model_path)
        fingerprint = fingerprint_local_model_path(model_path)
        if fingerprint != args.model_fingerprint:
            raise ValueError("speaker model fingerprint does not match")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch  # type: ignore[import-not-found]
        import torchaudio  # type: ignore[import-not-found]

        try:
            from speechbrain.inference.speaker import (  # type: ignore[import-not-found]
                EncoderClassifier,
            )
        except ImportError:
            from speechbrain.pretrained import (  # type: ignore[import-not-found,no-redef]
                EncoderClassifier,
            )

        classifier = EncoderClassifier.from_hparams(
            source=str(model_path),
            savedir=str(model_path),
            run_opts={"device": args.device},
        )
        self.args = args
        self.classifier = classifier
        self.torch = torch
        self.torchaudio = torchaudio
        self.backend_version = _package_version("speechbrain")

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        audio_path = Path(str(request["audio_path"])).expanduser().resolve(strict=True)
        waveform, sample_rate = self.torchaudio.load(str(audio_path))
        source_channels = int(waveform.shape[0]) if waveform.ndim == 2 else 0
        source_sample_count = int(waveform.shape[1]) if waveform.ndim == 2 else 0
        waveform, preprocessing = _prepare_model_input(
            waveform,
            int(sample_rate),
            torch=self.torch,
            torchaudio=self.torchaudio,
        )
        with self.torch.inference_mode():
            encoded = self.classifier.encode_batch(waveform)
        values = encoded.detach().to("cpu").float().numpy().reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("speaker model returned an invalid embedding")
        return {
            "status": "available",
            "model_identifier": self.args.model_identifier,
            "model_fingerprint": self.args.model_fingerprint,
            "embedding": values.tolist(),
            "dimension": int(values.size),
            "dtype": "float32",
            "backend_metadata": {
                "backend_name": "speechbrain_ecapa_voxceleb",
                "backend_version": self.backend_version,
                "source_sample_rate_hz": int(sample_rate),
                "source_channels": source_channels,
                "source_sample_count": source_sample_count,
                "source_duration_seconds": float(source_sample_count / sample_rate),
                "model_input_sample_rate_hz": 16000,
                "model_input_channels": 1,
                "model_input_sample_count": int(waveform.shape[1]),
                "resampling_policy_version": preprocessing,
                "embedding_dimension": int(values.size),
            },
        }


def _emit(response: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    sys.stdout.write("\n")
    sys.stdout.flush()


def main() -> int:
    args = _parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        worker = SpeechBrainWorker(args)
    for line in sys.stdin:
        request_id = "unknown"
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("embedding worker request must be an object")
            request_id = str(request.get("request_id", "unknown"))
            if request.get("operation") == "shutdown":
                _emit({"request_id": request_id, "status": "shutdown"})
                return 0
            if request.get("operation") != "speaker_embedding":
                raise ValueError("unsupported speaker embedding operation")
            with contextlib.redirect_stdout(sys.stderr):
                response = worker.process(request)
            _emit({"request_id": request_id, **response})
        except Exception as exc:  # noqa: BLE001 - isolate one inference request
            traceback.print_exc(file=sys.stderr)
            _emit(
                {
                    "request_id": request_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "model_identifier": args.model_identifier,
                    "model_fingerprint": args.model_fingerprint,
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
