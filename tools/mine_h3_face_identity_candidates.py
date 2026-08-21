#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.h3.audio_backends import (
    PersistentSubprocessEmbeddingBackend,
    fingerprint_local_model_path,
)
from r2v_data_v2.h3.face_identity_mining import mine_face_identity_candidates


def _python_executable(value: str) -> str:
    path = Path(value).expanduser()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"face embedding Python executable is missing: {value}")
    return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine read-only Visual face identity candidates for H3 calibration"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--face-python", required=True)
    parser.add_argument("--face-model-root", type=Path, required=True)
    parser.add_argument("--face-model-name", required=True)
    parser.add_argument("--face-model-identifier", required=True)
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
    model_root = args.face_model_root.expanduser().resolve(strict=True)
    face_pack = (model_root / "models" / args.face_model_name).resolve(strict=True)
    face_pack.relative_to(model_root)
    fingerprint = fingerprint_local_model_path(face_pack)
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if args.cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    backend = PersistentSubprocessEmbeddingBackend(
        executable=[
            _python_executable(args.face_python),
            str(repository / "tools" / "run_h3_face_embedding_worker.py"),
            "--model-root",
            str(model_root),
            "--model-name",
            args.face_model_name,
            "--model-identifier",
            args.face_model_identifier,
            "--model-fingerprint",
            fingerprint,
            "--device",
            args.device,
        ],
        model_identifier=args.face_model_identifier,
        checkpoint_sha256=fingerprint,
        timeout_seconds=args.timeout_seconds,
        diagnostics_root=diagnostics,
        environment=environment,
    )
    try:
        summary = mine_face_identity_candidates(
            run_root=args.run_root,
            output_root=output_root,
            face_backend=backend,
            top_k=args.top_k,
            overwrite=args.overwrite,
        )
    finally:
        backend.close()
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
