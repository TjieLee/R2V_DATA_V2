#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from pathlib import Path


def _fingerprint_local_path(path: Path) -> str:
    source = path.expanduser().resolve(strict=True)
    files = (
        [source]
        if source.is_file()
        else sorted(item for item in source.rglob("*") if item.is_file())
    )
    if not files:
        raise ValueError("local DiariZen model cache contains no files")
    digest = hashlib.sha256()
    for item in files:
        relative = (
            item.name if source.is_file() else item.relative_to(source).as_posix()
        )
        digest.update(relative.encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent local DiariZen worker")
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument("--model-identifier", required=True)
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--device", required=True)
    return parser


def _emit(value: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    sys.stdout.flush()


def _load_pipeline(arguments: argparse.Namespace) -> tuple[object, object]:
    code_root = arguments.code_root.expanduser().resolve(strict=True)
    model_cache = arguments.model_cache.expanduser().resolve(strict=True)
    if not code_root.is_dir() or not model_cache.is_dir():
        raise ValueError("DiariZen code root and model cache must be local directories")
    fingerprint = _fingerprint_local_path(model_cache)
    if fingerprint != arguments.model_fingerprint:
        raise ValueError("local DiariZen model fingerprint changed")
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    device = torch.device(arguments.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DiariZen CUDA device requested but CUDA is unavailable")
        if device.index is None or device.index >= torch.cuda.device_count():
            raise RuntimeError("requested DiariZen CUDA device is unavailable")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("DiariZen device must be cpu or cuda:N")

    from diarizen.pipelines.inference import DiariZenPipeline

    # Official DiariZen emits progress messages with print(); keep JSONL stdout clean.
    with contextlib.redirect_stdout(sys.stderr):
        pipeline = DiariZenPipeline.from_pretrained(
            arguments.model_identifier,
            cache_dir=str(model_cache),
        )
        pipeline.to(device)
    return pipeline, device


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        pipeline, device = _load_pipeline(arguments)
    except Exception as exc:  # noqa: BLE001 - startup must fail closed with diagnostics
        print(f"DiariZen startup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
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
            "model_identifier": arguments.model_identifier,
            "model_fingerprint": arguments.model_fingerprint,
            "device": str(device),
        }
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = "unknown"
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("worker request must be an object")
            request_id = str(request.get("request_id") or "unknown")
            operation = request.get("operation")
            if operation == "shutdown":
                _emit({"request_id": request_id, "status": "shutdown"})
                return 0
            if operation != "diarize":
                raise ValueError("unsupported DiariZen worker operation")
            if request.get("model_identifier") != arguments.model_identifier:
                raise ValueError("DiariZen request model identifier mismatch")
            clip_uid = str(request.get("clip_uid") or "")
            audio_path = (
                Path(str(request.get("audio_path") or ""))
                .expanduser()
                .resolve(strict=True)
            )
            if not clip_uid or not audio_path.is_file():
                raise ValueError("DiariZen request media is invalid")
            with contextlib.redirect_stdout(sys.stderr):
                result = pipeline(str(audio_path), sess_name=clip_uid)
            segments = [
                {
                    "start_time": float(turn.start),
                    "end_time": float(turn.end),
                    "speaker_label": str(speaker),
                }
                for turn, _, speaker in result.itertracks(yield_label=True)
            ]
            segments.sort(
                key=lambda item: (
                    item["start_time"],
                    item["end_time"],
                    item["speaker_label"],
                )
            )
            _emit(
                {
                    "request_id": request_id,
                    "status": "ready",
                    "model_identifier": arguments.model_identifier,
                    "segments": segments,
                    "backend_metadata": {
                        "backend": "diarizen_official_pipeline",
                        "device": str(device),
                        "input_preprocessing": (
                            "official_torchaudio_first_channel_passthrough_v1"
                        ),
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001 - isolate one clip request
            print(
                f"DiariZen request {request_id} failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
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
