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

from r2v_data_v2.h3.audio_backends import fingerprint_local_model_path
from r2v_data_v2.h3.diarization_binding import (
    DEFAULT_DIARIZEN_DEVICE,
    DEFAULT_DIARIZEN_MODEL_IDENTIFIER,
    DEFAULT_DIARIZEN_TIMEOUT_SECONDS,
    DIARIZATION_PREPROCESSING_VERSION,
    DIARIZATION_REQUEST_VERSION,
    DiarizationBackendProvenance,
    PersistentDiariZenBackend,
    build_diarization_inventory,
    diarization_output_root,
    run_diarization_binding_pilot,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the read-only-input H3 DiariZen speaker binding pilot",
    )
    parser.add_argument("--audio-run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot20", "production"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"missing DiariZen runtime environment variable: {name}")
    return value


def _configuration_fingerprint(
    *,
    model_identifier: str,
    model_fingerprint: str,
    requested_device: str,
) -> str:
    import hashlib

    payload = json.dumps(
        {
            "backend": "diarizen_official_pipeline",
            "model_identifier": model_identifier,
            "model_fingerprint": model_fingerprint,
            "requested_device": requested_device,
            "request_contract_version": DIARIZATION_REQUEST_VERSION,
            "input_preprocessing": DIARIZATION_PREPROCESSING_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _runtime_backend(
    *,
    output_root: Path,
) -> tuple[PersistentDiariZenBackend, Path]:
    python_value = _required_environment("DIARIZEN_PYTHON")
    python_path = Path(python_value).expanduser()
    if not python_path.exists() or not python_path.is_file():
        raise FileNotFoundError("DIARIZEN_PYTHON is not an existing executable")
    code_root = (
        Path(_required_environment("DIARIZEN_CODE_ROOT"))
        .expanduser()
        .resolve(strict=True)
    )
    model_cache = (
        Path(_required_environment("DIARIZEN_MODEL_PATH"))
        .expanduser()
        .resolve(strict=True)
    )
    if not code_root.is_dir() or not model_cache.is_dir():
        raise ValueError("DiariZen code/model paths must be local directories")
    model_identifier = os.environ.get(
        "DIARIZEN_MODEL_IDENTIFIER", DEFAULT_DIARIZEN_MODEL_IDENTIFIER
    ).strip()
    requested_device = os.environ.get(
        "DIARIZEN_DEVICE", DEFAULT_DIARIZEN_DEVICE
    ).strip()
    if not model_identifier:
        raise ValueError("DIARIZEN_MODEL_IDENTIFIER must not be empty")
    if not requested_device.startswith("cuda:"):
        raise ValueError("formal DiariZen pilot requires an explicit cuda:N device")
    try:
        cuda_index = int(requested_device.split(":", 1)[1])
    except ValueError as exc:
        raise ValueError("DIARIZEN_DEVICE must use cuda:N") from exc
    if cuda_index < 0:
        raise ValueError("DIARIZEN_DEVICE index must be non-negative")
    try:
        timeout_seconds = float(
            os.environ.get(
                "DIARIZEN_TIMEOUT_SECONDS",
                str(DEFAULT_DIARIZEN_TIMEOUT_SECONDS),
            )
        )
    except ValueError as exc:
        raise ValueError("DIARIZEN_TIMEOUT_SECONDS must be numeric") from exc
    if timeout_seconds <= 0:
        raise ValueError("DIARIZEN_TIMEOUT_SECONDS must be positive")
    model_fingerprint = fingerprint_local_model_path(model_cache)
    provenance = DiarizationBackendProvenance(
        backend="diarizen_official_pipeline",
        model_identifier=model_identifier,
        model_fingerprint=model_fingerprint,
        configuration_fingerprint=_configuration_fingerprint(
            model_identifier=model_identifier,
            model_fingerprint=model_fingerprint,
            requested_device=requested_device,
        ),
    )
    hidden_diagnostics = output_root.parent / (
        f".{output_root.name}.worker-{uuid.uuid4().hex}"
    )
    environment = {
        "CUDA_VISIBLE_DEVICES": str(cuda_index),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    backend = PersistentDiariZenBackend(
        executable=[
            str(python_path),
            str(REPOSITORY_ROOT / "tools" / "diarizen_worker.py"),
            "--code-root",
            str(code_root),
            "--model-cache",
            str(model_cache),
            "--model-identifier",
            model_identifier,
            "--model-fingerprint",
            model_fingerprint,
            "--device",
            "cuda:0",
        ],
        provenance=provenance,
        timeout_seconds=timeout_seconds,
        diagnostics_root=hidden_diagnostics,
        environment=environment,
    )
    return backend, hidden_diagnostics


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    root = arguments.audio_run_root.expanduser().resolve(strict=True)
    inventory = build_diarization_inventory(
        audio_run_root=root,
        mode=arguments.mode,
    )
    output_root = diarization_output_root(root, mode=arguments.mode)
    plan: dict[str, object] = {
        "mode": arguments.mode,
        "source_target_count": inventory.source_target_count,
        "selected_target_count": inventory.selected_target_count,
        "source_asr_inventory_fingerprint": (
            inventory.source_asr_inventory_fingerprint
        ),
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "output_root": str(output_root),
        "parent_quota_applied": False,
        "donor_media_used": False,
        "cross_pair_jobs_created": 0,
        "production_blocked": inventory.production_blocked,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return plan
    if arguments.mode == "production":
        raise ValueError("production_blocked_pending_diarization_binding_calibration")

    backend, diagnostics_root = _runtime_backend(output_root=output_root)
    try:
        with backend:
            summary = run_diarization_binding_pilot(
                inventory=inventory,
                output_root=output_root,
                backend=backend,
                overwrite=arguments.overwrite,
            )
    finally:
        if diagnostics_root.exists():
            shutil.rmtree(diagnostics_root)
    result = {
        **plan,
        "stage_status": "completed",
        "summary": summary.model_dump(mode="json"),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
