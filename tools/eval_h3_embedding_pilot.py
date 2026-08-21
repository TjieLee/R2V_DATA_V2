#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.h3.audio_backends import (
    PersistentSubprocessEmbeddingBackend,
    fingerprint_local_model_path,
)
from r2v_data_v2.h3.embedding_pilot import (
    load_embedding_pilot_inputs,
    run_embedding_pilot,
)


def _python_executable(value: str) -> str:
    path = Path(value).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"embedding Python executable is missing: {value}")
    return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build read-only H3 face/speaker embedding retrieval diagnostics"
    )
    parser.add_argument("--audio-pilot-root", type=Path, required=True)
    parser.add_argument("--primary-voice-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--face-python", required=True)
    parser.add_argument("--face-model-root", type=Path, required=True)
    parser.add_argument("--face-model-name", required=True)
    parser.add_argument("--face-model-identifier", required=True)
    parser.add_argument("--speaker-python", required=True)
    parser.add_argument("--speaker-model-path", type=Path, required=True)
    parser.add_argument(
        "--speaker-model-identifier",
        default="speechbrain/spkrec-ecapa-voxceleb",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = Path(__file__).resolve().parents[1]
    output_root = args.output_root.expanduser().resolve(strict=False)
    diagnostics = output_root.parent / f".{output_root.name}.worker_diagnostics"
    face_pack = (
        args.face_model_root.expanduser().resolve(strict=True)
        / "models"
        / args.face_model_name
    ).resolve(strict=True)
    face_pack.relative_to(args.face_model_root.expanduser().resolve(strict=True))
    speaker_model = args.speaker_model_path.expanduser().resolve(strict=True)
    face_fingerprint = fingerprint_local_model_path(face_pack)
    speaker_fingerprint = fingerprint_local_model_path(speaker_model)
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if args.cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    face_backend = PersistentSubprocessEmbeddingBackend(
        executable=[
            _python_executable(args.face_python),
            str(repository / "tools" / "run_h3_face_embedding_worker.py"),
            "--model-root",
            str(args.face_model_root),
            "--model-name",
            args.face_model_name,
            "--model-identifier",
            args.face_model_identifier,
            "--model-fingerprint",
            face_fingerprint,
            "--device",
            args.device,
        ],
        model_identifier=args.face_model_identifier,
        checkpoint_sha256=face_fingerprint,
        timeout_seconds=args.timeout_seconds,
        diagnostics_root=diagnostics / "face",
        environment=environment,
    )
    speaker_backend = PersistentSubprocessEmbeddingBackend(
        executable=[
            _python_executable(args.speaker_python),
            str(repository / "tools" / "run_h3_speaker_embedding_worker.py"),
            "--model-path",
            str(speaker_model),
            "--model-identifier",
            args.speaker_model_identifier,
            "--model-fingerprint",
            speaker_fingerprint,
            "--device",
            args.device,
        ],
        model_identifier=args.speaker_model_identifier,
        checkpoint_sha256=speaker_fingerprint,
        timeout_seconds=args.timeout_seconds,
        diagnostics_root=diagnostics / "speaker",
        environment=environment,
    )
    inputs = load_embedding_pilot_inputs(
        audio_pilot_root=args.audio_pilot_root,
        primary_voice_root=args.primary_voice_root,
    )
    try:
        summary = run_embedding_pilot(
            inputs=inputs,
            output_root=output_root,
            face_backend=face_backend,
            speaker_backend=speaker_backend,
            top_k=args.top_k,
            overwrite=args.overwrite,
        )
    finally:
        face_backend.close()
        speaker_backend.close()
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
