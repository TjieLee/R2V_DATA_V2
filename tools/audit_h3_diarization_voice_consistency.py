#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.audio_backends import (
    PersistentSubprocessEmbeddingBackend,
    fingerprint_local_model_path,
)
from r2v_data_v2.h3.diarization_voice_consistency_audit import (
    DEFAULT_OUTPUT_DIRECTORY,
    SPEAKER_MODEL_IDENTIFIER,
    run_diarization_voice_consistency_audit,
)
from r2v_data_v2.h3.jea_audio_production import jea_production_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit DiariZen propagation against target ECAPA voice references",
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--speaker-python")
    parser.add_argument("--speaker-model-path", type=Path)
    parser.add_argument(
        "--speaker-model-identifier",
        default=SPEAKER_MODEL_IDENTIFIER,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _text_argument(explicit: str | None, environment_name: str) -> str:
    value = explicit or os.environ.get(environment_name)
    if value is None or not value.strip():
        raise ValueError(f"missing runtime input: {environment_name}")
    return value


def _path_argument(explicit: Path | None, environment_name: str) -> Path:
    if explicit is not None:
        return explicit
    value = os.environ.get(environment_name)
    if value is None or not value.strip():
        raise ValueError(f"missing runtime input: {environment_name}")
    return Path(value)


def _python_executable(value: str) -> str:
    path = Path(value).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"speaker Python executable is missing: {value}")
    return str(path)


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    if arguments.timeout_seconds <= 0:
        raise ValueError("speaker embedding timeout must be positive")
    production_paths = jea_production_paths(arguments.audio_production_root)
    output_root = (
        arguments.output_root.expanduser().resolve(strict=False)
        if arguments.output_root is not None
        else production_paths.root / DEFAULT_OUTPUT_DIRECTORY
    )
    speaker_python = _text_argument(
        arguments.speaker_python,
        "SPEAKER_EMBEDDING_PYTHON",
    )
    speaker_model = _path_argument(
        arguments.speaker_model_path,
        "SPEAKER_MODEL_PATH",
    ).expanduser().resolve(strict=True)
    model_fingerprint = fingerprint_local_model_path(speaker_model)
    environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    cuda_visible = arguments.cuda_visible_devices or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if cuda_visible is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible
    diagnostics = output_root.parent / (
        f".{output_root.name}.worker-{uuid.uuid4().hex}"
    )
    backend = PersistentSubprocessEmbeddingBackend(
        executable=[
            _python_executable(speaker_python),
            str(REPOSITORY_ROOT / "tools" / "run_h3_speaker_embedding_worker.py"),
            "--model-path",
            str(speaker_model),
            "--model-identifier",
            arguments.speaker_model_identifier,
            "--model-fingerprint",
            model_fingerprint,
            "--device",
            arguments.device,
        ],
        model_identifier=arguments.speaker_model_identifier,
        checkpoint_sha256=model_fingerprint,
        timeout_seconds=arguments.timeout_seconds,
        diagnostics_root=diagnostics,
        environment=environment,
    )
    try:
        with backend:
            summary = run_diarization_voice_consistency_audit(
                audio_production_root=arguments.audio_production_root,
                speaker_backend=backend,
                output_root=output_root,
                overwrite=arguments.overwrite,
                ffmpeg=arguments.ffmpeg,
                ffprobe=arguments.ffprobe,
            )
    finally:
        if diagnostics.exists():
            shutil.rmtree(diagnostics)
    result = {
        "output_root": str(output_root),
        "audited_segment_count": summary.audited_segment_count,
        "skipped_segment_count": summary.skipped_segment_count,
        "direct_anchor_present_count": summary.direct_anchor_present_count,
        "cluster_propagated_only_count": summary.cluster_propagated_only_count,
        "review_candidate_count": summary.review_candidate_count,
        "similarity_threshold_applied": False,
        "binding_modified": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
