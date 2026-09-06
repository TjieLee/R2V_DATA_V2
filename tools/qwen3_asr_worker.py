#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent isolated Qwen3-ASR worker")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        required=True,
    )
    parser.add_argument("--max-inference-batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser


def _emit(value: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    sys.stdout.flush()


def _load_model(arguments: argparse.Namespace) -> tuple[object, object]:
    model_path = arguments.model_path.expanduser().resolve(strict=True)
    if not model_path.is_dir():
        raise ValueError("Qwen3-ASR model path must be a local directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from qwen_asr import Qwen3ASRModel

    device = str(arguments.device)
    if device.partition(":")[0].lower() == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-ASR CUDA device requested but CUDA is unavailable")
        try:
            cuda_index = int(device.partition(":")[2])
        except ValueError as exc:
            raise ValueError("Qwen3-ASR CUDA device must use cuda:N") from exc
        if cuda_index < 0 or cuda_index >= torch.cuda.device_count():
            raise RuntimeError("requested Qwen3-ASR CUDA device is unavailable")
    dtype = getattr(torch, arguments.dtype)
    with contextlib.redirect_stdout(sys.stderr):
        model = Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=dtype,
            device_map=device,
            max_inference_batch_size=arguments.max_inference_batch_size,
            max_new_tokens=arguments.max_new_tokens,
            local_files_only=True,
        )
    return model, torch


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        model, _ = _load_model(arguments)
        import numpy as np
    except Exception as exc:  # noqa: BLE001 - startup must fail closed with diagnostics
        print(f"Qwen3-ASR startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        _emit(
            {
                "request_id": "startup",
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return 1

    _emit(
        {
            "request_id": "startup",
            "status": "ready",
            "model_path": str(arguments.model_path.expanduser().resolve(strict=True)),
            "device": arguments.device,
        }
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = "unknown"
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("Qwen3-ASR worker request must be an object")
            request_id = str(request.get("request_id") or "unknown")
            operation = request.get("operation")
            if operation == "shutdown":
                _emit({"request_id": request_id, "status": "shutdown"})
                return 0
            if operation != "transcribe":
                raise ValueError("unsupported Qwen3-ASR worker operation")
            sample_rate_hz = int(request.get("sample_rate_hz") or 0)
            sample_count = int(request.get("sample_count") or 0)
            payload = request.get("audio_f32le_base64")
            if sample_rate_hz != 16000 or sample_count <= 0 or not isinstance(payload, str):
                raise ValueError("Qwen3-ASR worker audio request is invalid")
            raw = base64.b64decode(payload.encode("ascii"), validate=True)
            waveform = np.frombuffer(raw, dtype="<f4").copy()
            if waveform.ndim != 1 or waveform.size != sample_count:
                raise ValueError("Qwen3-ASR worker sample count differs")
            if not np.isfinite(waveform).all():
                raise ValueError("Qwen3-ASR worker waveform is non-finite")
            with contextlib.redirect_stdout(sys.stderr):
                result = model.transcribe(
                    audio=(waveform, sample_rate_hz),
                    context="",
                    language=None,
                    return_time_stamps=False,
                )[0]
            _emit(
                {
                    "request_id": request_id,
                    "status": "ok",
                    "text": str(result.text),
                    "language": (
                        None if result.language is None else str(result.language)
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - isolate individual requests
            _emit(
                {
                    "request_id": request_id,
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
