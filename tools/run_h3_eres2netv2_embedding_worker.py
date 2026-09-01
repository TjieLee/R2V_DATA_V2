#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Protocol, TextIO

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.audio_backends import fingerprint_local_model_path

ERES2NETV2_MODEL_IDENTIFIER = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
ERES2NETV2_MODEL_REVISION = "v1.0.1"
ERES2NETV2_CHECKPOINT_NAME = "pretrained_eres2netv2.ckpt"
ERES2NETV2_MODEL_OBJECT = "speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2"
ERES2NETV2_MODEL_ARGS = {
    "feat_dim": 80,
    "embedding_size": 192,
    "baseWidth": 26,
    "scale": 2,
    "expansion": 2,
}


class _EmbeddingWorker(Protocol):
    def process(self, request: dict[str, Any]) -> dict[str, Any]: ...


def validate_speakerlab_code_root(code_root: Path) -> Path:
    root = code_root.expanduser().resolve(strict=True)
    required = (
        "speakerlab/process/processor.py",
        "speakerlab/utils/builder.py",
        "speakerlab/models/eres2net/ERes2NetV2.py",
    )
    if not root.is_dir() or any(not (root / value).is_file() for value in required):
        raise FileNotFoundError(
            "SpeakerLab source root lacks the official ERes2NetV2 inference modules"
        )
    return root


def resolve_eres2netv2_checkpoint(model_path: Path) -> tuple[Path, Path]:
    source = model_path.expanduser().resolve(strict=True)
    checkpoint = source / ERES2NETV2_CHECKPOINT_NAME if source.is_dir() else source
    if not checkpoint.is_file() or checkpoint.name != ERES2NETV2_CHECKPOINT_NAME:
        raise FileNotFoundError(
            "ERes2NetV2 model path must resolve to pretrained_eres2netv2.ckpt"
        )
    return source, checkpoint


def fingerprint_speakerlab_source(code_root: Path) -> str:
    root = validate_speakerlab_code_root(code_root)
    files = sorted((root / "speakerlab").rglob("*.py"))
    if not files:
        raise ValueError("SpeakerLab source contains no Python modules")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Persistent offline SpeakerLab ERes2NetV2 embedding worker"
    )
    parser.add_argument("--speakerlab-code-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-identifier", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _assert_module_owned_by(module_object: object, root: Path) -> None:
    source = Path(inspect.getfile(module_object)).resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("SpeakerLab import did not come from explicit code root") from exc


class ERes2NetV2Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        if args.model_identifier != ERES2NETV2_MODEL_IDENTIFIER:
            raise ValueError("worker only supports the official ERes2NetV2 common model")
        code_root = validate_speakerlab_code_root(args.speakerlab_code_root)
        model_source, checkpoint = resolve_eres2netv2_checkpoint(args.model_path)
        fingerprint = fingerprint_local_model_path(model_source)
        if fingerprint != args.model_fingerprint:
            raise ValueError("ERes2NetV2 model fingerprint does not match")

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
        sys.path.insert(0, str(code_root))

        import torch  # type: ignore[import-not-found]
        import torchaudio  # type: ignore[import-not-found]
        from speakerlab.process.processor import FBank  # type: ignore[import-not-found]
        from speakerlab.utils.builder import (  # type: ignore[import-not-found]
            dynamic_import,
        )

        model_type = dynamic_import(ERES2NETV2_MODEL_OBJECT)
        _assert_module_owned_by(FBank, code_root)
        _assert_module_owned_by(model_type, code_root)
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ERes2NetV2 CUDA device was requested but unavailable")

        state = torch.load(checkpoint, map_location="cpu")
        model = model_type(**ERES2NETV2_MODEL_ARGS)
        model.load_state_dict(state, strict=True)
        model.to(device)
        model.eval()

        self.args = args
        self.checkpoint = checkpoint
        self.code_root = code_root
        self.source_fingerprint = fingerprint_speakerlab_source(code_root)
        self.feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        self.model = model
        self.device = device
        self.torch = torch
        self.torchaudio = torchaudio

    def process(self, request: dict[str, Any]) -> dict[str, Any]:
        audio_path = Path(str(request["audio_path"])).expanduser().resolve(strict=True)
        waveform, sample_rate = self.torchaudio.load(str(audio_path))
        if waveform.ndim != 2 or waveform.shape[0] != 1 or sample_rate != 16000:
            raise ValueError("ERes2NetV2 input must be 16 kHz mono audio")
        if waveform.shape[1] <= 0 or not self.torch.isfinite(waveform).all().item():
            raise ValueError("ERes2NetV2 waveform must be finite and non-empty")
        features = self.feature_extractor(waveform).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            encoded = self.model(features)
        values = encoded.detach().squeeze(0).to("cpu").float().numpy().reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("ERes2NetV2 returned an invalid embedding")
        return {
            "status": "available",
            "model_identifier": self.args.model_identifier,
            "model_fingerprint": self.args.model_fingerprint,
            "embedding": values.tolist(),
            "dimension": int(values.size),
            "dtype": "float32",
            "backend_metadata": {
                "backend_name": "speakerlab_eres2netv2",
                "backend_revision": ERES2NETV2_MODEL_REVISION,
                "speakerlab_source_root": str(self.code_root),
                "speakerlab_source_fingerprint": self.source_fingerprint,
                "checkpoint_path": str(self.checkpoint),
                "torch_version": _package_version("torch"),
                "torchaudio_version": _package_version("torchaudio"),
                "source_sample_rate_hz": int(sample_rate),
                "source_sample_count": int(waveform.shape[1]),
                "source_duration_seconds": float(waveform.shape[1] / sample_rate),
                "embedding_dimension": int(values.size),
            },
        }


def _emit(stream: TextIO, response: dict[str, Any]) -> None:
    stream.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))
    stream.write("\n")
    stream.flush()


def serve_jsonl_requests(
    worker: _EmbeddingWorker,
    *,
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
    model_identifier: str,
    model_fingerprint: str,
) -> int:
    for line in input_stream:
        request_id = "unknown"
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("embedding worker request must be an object")
            request_id = str(request.get("request_id", "unknown"))
            if request.get("operation") == "shutdown":
                _emit(output_stream, {"request_id": request_id, "status": "shutdown"})
                return 0
            if request.get("operation") != "speaker_embedding":
                raise ValueError("unsupported ERes2NetV2 embedding operation")
            with contextlib.redirect_stdout(error_stream):
                response = worker.process(request)
            _emit(output_stream, {"request_id": request_id, **response})
        except Exception as exc:  # noqa: BLE001 - isolate one inference request
            traceback.print_exc(file=error_stream)
            _emit(
                output_stream,
                {
                    "request_id": request_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "model_identifier": model_identifier,
                    "model_fingerprint": model_fingerprint,
                },
            )
    return 0


def main() -> int:
    args = _parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        worker = ERes2NetV2Worker(args)
    return serve_jsonl_requests(
        worker,
        input_stream=sys.stdin,
        output_stream=sys.stdout,
        error_stream=sys.stderr,
        model_identifier=args.model_identifier,
        model_fingerprint=args.model_fingerprint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
