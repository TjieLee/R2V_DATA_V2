#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal


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
    parser.add_argument(
        "--input-profile",
        choices=("canonical_32k_stereo", "legacy_16k_mono"),
        required=True,
    )
    return parser


def _emit(value: dict[str, object]) -> None:
    sys.stdout.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    sys.stdout.flush()


_CANONICAL_RESAMPLE_POLICY = (
    "h3_diarizen_torchaudio_kaiser_32k_stereo_to_16k_mono_v1"
)
_LEGACY_PASSTHROUGH_POLICY = "h3_diarizen_native_16k_mono_passthrough_v1"

InputProfile = Literal["canonical_32k_stereo", "legacy_16k_mono"]


def _prepare_analysis_audio(
    *,
    source: Path,
    destination: Path,
    torch: Any,
    torchaudio: Any,
    input_profile: InputProfile = "canonical_32k_stereo",
) -> Path:
    waveform, sample_rate = torchaudio.load(str(source))
    if waveform.ndim != 2 or waveform.shape[1] <= 0:
        raise ValueError("DiariZen source must be channel-first non-empty audio")
    if not torch.isfinite(waveform).all().item():
        raise ValueError("DiariZen source must contain only finite samples")
    if input_profile == "legacy_16k_mono":
        if waveform.shape[0] != 1 or sample_rate != 16000:
            raise ValueError("legacy DiariZen source must be 16 kHz mono")
        return source
    if waveform.shape[0] != 2 or sample_rate != 32000:
        raise ValueError("DiariZen canonical source must be 32 kHz stereo")
    mono = waveform.mean(dim=0, keepdim=True)
    analysis = torchaudio.functional.resample(
        mono,
        orig_freq=32000,
        new_freq=16000,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(
        str(destination),
        analysis,
        16000,
        encoding="PCM_S",
        bits_per_sample=16,
    )
    return destination


@contextlib.contextmanager
def _all_inactive_reconstruct_guard(
    pipeline: object,
    *,
    numpy: Any,
) -> Any:
    instance_attributes = vars(pipeline)
    had_instance_override = "reconstruct" in instance_attributes
    previous_instance_override = instance_attributes.get("reconstruct")
    original = pipeline.reconstruct
    state = {"applied": False}

    def guarded(*args: object, **kwargs: object) -> object:
        bound = inspect.signature(original).bind(*args, **kwargs)
        hard_clusters = bound.arguments.get("hard_clusters")
        segmentations = bound.arguments.get("segmentations")
        if hard_clusters is not None and segmentations is not None:
            clusters = numpy.asarray(hard_clusters)
            segmentation_data = numpy.asarray(segmentations.data)
            inactive = numpy.sum(segmentation_data, axis=1) == 0
            if (
                clusters.size > 0
                and inactive.shape == clusters.shape
                and bool(numpy.all(clusters == -2))
                and bool(numpy.all(inactive))
            ):
                working_clusters = numpy.array(hard_clusters, copy=True)
                working_clusters[...] = 0
                bound.arguments["hard_clusters"] = working_clusters
                state["applied"] = True
        return original(*bound.args, **bound.kwargs)

    pipeline.reconstruct = guarded
    try:
        yield state
    finally:
        if had_instance_override:
            pipeline.reconstruct = previous_instance_override
        else:
            del pipeline.reconstruct


def _load_pipeline(arguments: argparse.Namespace) -> tuple[object, object, object, object]:
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
    import torchaudio

    return pipeline, device, torch, torchaudio


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        pipeline, device, torch, torchaudio = _load_pipeline(arguments)
        import numpy as np
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
            if request.get("input_profile") != arguments.input_profile:
                raise ValueError("DiariZen request input profile mismatch")
            clip_uid = str(request.get("clip_uid") or "")
            audio_path = (
                Path(str(request.get("audio_path") or ""))
                .expanduser()
                .resolve(strict=True)
            )
            if not clip_uid or not audio_path.is_file():
                raise ValueError("DiariZen request media is invalid")
            with tempfile.TemporaryDirectory(prefix="h3-diarizen-analysis-") as work:
                analysis_path = Path(work) / "analysis_16k_mono.wav"
                model_audio_path = _prepare_analysis_audio(
                    source=audio_path,
                    destination=analysis_path,
                    torch=torch,
                    torchaudio=torchaudio,
                    input_profile=arguments.input_profile,
                )
                with contextlib.ExitStack() as stack:
                    guard_state = stack.enter_context(
                        _all_inactive_reconstruct_guard(pipeline, numpy=np)
                    )
                    stack.enter_context(contextlib.redirect_stdout(sys.stderr))
                    result = pipeline(str(model_audio_path), sess_name=clip_uid)
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
                        "input_profile": arguments.input_profile,
                        "input_preprocessing": (
                            _CANONICAL_RESAMPLE_POLICY
                            if arguments.input_profile == "canonical_32k_stereo"
                            else _LEGACY_PASSTHROUGH_POLICY
                        ),
                        "source_sample_rate_hz": (
                            32000
                            if arguments.input_profile == "canonical_32k_stereo"
                            else 16000
                        ),
                        "source_channels": (
                            2
                            if arguments.input_profile == "canonical_32k_stereo"
                            else 1
                        ),
                        "model_input_sample_rate_hz": 16000,
                        "model_input_channels": 1,
                        "all_inactive_reconstruction_guard_applied": guard_state[
                            "applied"
                        ],
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
