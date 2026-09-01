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
from r2v_data_v2.h3.eres2netv2_voice_consistency_shadow import (
    DEFAULT_OUTPUT_DIRECTORY,
    run_eres2netv2_voice_consistency_shadow,
)
from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from tools.run_h3_eres2netv2_embedding_worker import (
    ERES2NETV2_MODEL_IDENTIFIER,
    resolve_eres2netv2_checkpoint,
    validate_speakerlab_code_root,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare ERes2NetV2 and ECAPA against voice-consistency labels",
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--speakerlab-code-root", type=Path)
    parser.add_argument("--eres2netv2-python")
    parser.add_argument("--eres2netv2-model-path", type=Path)
    parser.add_argument("--model-identifier", default=ERES2NETV2_MODEL_IDENTIFIER)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
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
        raise FileNotFoundError(f"ERes2NetV2 Python executable is missing: {value}")
    return str(path)


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    if arguments.timeout_seconds <= 0:
        raise ValueError("ERes2NetV2 embedding timeout must be positive")
    paths = jea_production_paths(arguments.audio_production_root)
    audit_root = (
        arguments.audit_root.expanduser().resolve(strict=True)
        if arguments.audit_root is not None
        else paths.root / "diarization_voice_consistency_audit_v1"
    )
    output_root = (
        arguments.output_root.expanduser().resolve(strict=False)
        if arguments.output_root is not None
        else audit_root / DEFAULT_OUTPUT_DIRECTORY
    )
    speakerlab_root = validate_speakerlab_code_root(
        _path_argument(arguments.speakerlab_code_root, "SPEAKERLAB_CODE_ROOT")
    )
    model_source, _ = resolve_eres2netv2_checkpoint(
        _path_argument(arguments.eres2netv2_model_path, "ERES2NETV2_MODEL_PATH")
    )
    model_fingerprint = fingerprint_local_model_path(model_source)
    python = _python_executable(
        _text_argument(arguments.eres2netv2_python, "ERES2NETV2_PYTHON")
    )
    environment = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MODELSCOPE_OFFLINE": "1",
    }
    cuda_visible = arguments.cuda_visible_devices or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if cuda_visible is not None:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible
    diagnostics = output_root.parent / f".{output_root.name}.worker-{uuid.uuid4().hex}"
    backend = PersistentSubprocessEmbeddingBackend(
        executable=[
            python,
            str(REPOSITORY_ROOT / "tools/run_h3_eres2netv2_embedding_worker.py"),
            "--speakerlab-code-root",
            str(speakerlab_root),
            "--model-path",
            str(model_source),
            "--model-identifier",
            arguments.model_identifier,
            "--model-fingerprint",
            model_fingerprint,
            "--device",
            arguments.device,
        ],
        model_identifier=arguments.model_identifier,
        checkpoint_sha256=model_fingerprint,
        timeout_seconds=arguments.timeout_seconds,
        diagnostics_root=diagnostics,
        environment=environment,
    )
    try:
        with backend:
            summary = run_eres2netv2_voice_consistency_shadow(
                audit_root=audit_root,
                speaker_backend=backend,
                output_root=output_root,
                overwrite=arguments.overwrite,
            )
    finally:
        if diagnostics.exists():
            shutil.rmtree(diagnostics)
    result = {
        "output_root": str(output_root),
        "current_annotation_count": summary.current_annotation_count,
        "evaluated_record_count": summary.evaluated_record_count,
        "error_count": summary.error_count,
        "uncertain_count": summary.uncertain_count,
        "model_call_count": summary.model_call_count,
        "production_threshold_applied": False,
        "binding_modified": False,
        "production_artifacts_modified": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
