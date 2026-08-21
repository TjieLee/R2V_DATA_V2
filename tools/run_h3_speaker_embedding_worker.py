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
        if waveform.ndim != 2 or waveform.shape[0] != 1 or sample_rate != 16000:
            raise ValueError("speaker input must be 16 kHz mono audio")
        if waveform.shape[1] <= 0 or not self.torch.isfinite(waveform).all().item():
            raise ValueError("speaker input waveform must be finite and non-empty")
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
                "source_sample_count": int(waveform.shape[1]),
                "source_duration_seconds": float(waveform.shape[1] / sample_rate),
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
