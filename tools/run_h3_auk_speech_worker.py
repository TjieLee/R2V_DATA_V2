#!/usr/bin/env python3
"""JSONL worker. Only this isolated process imports AuK and its dependencies."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dependencies(configuration: dict) -> None:
    for name, expected in configuration["dependency_files"].items():
        path = Path(name)
        if not path.is_absolute() or not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"AuK requires a non-empty local dependency: {name}")
        actual = file_hash(path)
        if actual != expected:
            raise ValueError(f"AuK dependency hash differs: {name}")


def run_worker(input_stream, output_stream) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def emit(payload):
        output_stream.write(json.dumps(payload, sort_keys=True) + "\n")
        output_stream.flush()

    try:
        configuration = json.loads(input_stream.readline())
        validate_dependencies(configuration)
        source_root = Path(configuration["code_root"]) / "src"
        sys.path.insert(0, str(source_root))
        with contextlib.redirect_stdout(sys.stderr):
            from auk.infer.infer_auk import AukInfer, save_audio

            engine = AukInfer(
                config_path=configuration["config_path"],
                ckpt_path=configuration["checkpoint_path"],
                device=configuration["device"],
                dtype=configuration["dtype"],
                qwen_path=configuration["qwen_path"],
            )
            if getattr(engine, "is_flash", False):
                raise ValueError("AuK speech v1 requires Base, not the Flash sampling recipe")
        emit({"request_id": "startup", "status": "ready"})
    except Exception as exc:  # noqa: BLE001 - report startup failure over JSONL
        emit(
            {
                "request_id": "startup",
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return
    for line in input_stream:
        request = json.loads(line)
        if request.get("operation") == "shutdown":
            return
        started = time.monotonic()
        try:
            source = Path(request["source_audio_path"])
            if not source.is_absolute() or not source.is_file():
                raise ValueError("AuK source must be a local audio file")
            if file_hash(source) != request["source_audio_sha256"]:
                raise ValueError("AuK source audio hash differs")
            duration = request["source_duration_seconds"]
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("AuK source duration must be positive and finite")
            output = Path(request["raw_output_path"])
            if not output.is_absolute() or output.exists():
                raise ValueError("AuK raw output must be a new absolute file")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": configuration["instruction"]},
                        {"type": "audio", "audio": str(source)},
                    ],
                }
            ]
            with contextlib.redirect_stdout(sys.stderr):
                audio, sample_rate = engine.generate(
                    messages,
                    gen_seconds=duration,
                    nfe=configuration["nfe"],
                    cfg_strength=configuration["cfg_strength"],
                    sway_sampling_coef=configuration["sway_sampling_coef"],
                    seed=configuration["seed"],
                )
                save_audio(audio, sample_rate, str(output))
            emit(
                {
                    "request_id": request["request_id"],
                    "status": "ok",
                    "model_runtime_seconds": time.monotonic() - started,
                }
            )
        except Exception as exc:  # noqa: BLE001 - preserve the persistent worker for other clips
            emit(
                {
                    "request_id": request["request_id"],
                    "status": "failed",
                    "model_runtime_seconds": time.monotonic() - started,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )


if __name__ == "__main__":
    run_worker(sys.stdin, sys.stdout)
