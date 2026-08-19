#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.asr_transcription import (
    DEFAULT_ASR_COMPUTE_TYPE,
    DEFAULT_ASR_DEVICE,
    DEFAULT_ASR_MODEL,
    FasterWhisperASRBackend,
    WhisperASRConfig,
)
from r2v_data_v2.h3.asr_v2_transcription import (
    ASRV2Inventory,
    ASRV2Summary,
    asr_v2_output_root,
    asr_v2_stage_is_complete,
    build_asr_v2_backend_provenance,
    build_asr_v2_inventory,
    regenerate_asr_v2_review,
    run_asr_v2_transcription,
)
from r2v_data_v2.h3.audio_backends import fingerprint_local_model_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe production DiariZen segments with Whisper-large-v3",
    )
    parser.add_argument("--audio-run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot20", "production"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--regenerate-review", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-fingerprint")
    parser.add_argument("--device")
    parser.add_argument("--compute-type")
    return parser


def _value(
    explicit: str | None,
    environment_name: str,
    *,
    default: str,
) -> str:
    value = explicit or os.environ.get(environment_name) or default
    if not value.strip():
        raise ValueError(f"missing ASR V2 runtime value: {environment_name}")
    return value.strip()


def _model_identity(arguments: argparse.Namespace) -> tuple[str, str | None]:
    explicit_fingerprint = arguments.model_fingerprint or os.environ.get(
        "ASR_MODEL_FINGERPRINT"
    )
    model_path = arguments.model_path
    if model_path is None:
        environment_path = os.environ.get("ASR_MODEL_PATH")
        if environment_path:
            model_path = Path(environment_path)
    if model_path is not None:
        resolved = model_path.expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("ASR_MODEL_PATH must be a local model directory")
        derived_fingerprint = fingerprint_local_model_path(resolved)
        if (
            explicit_fingerprint is not None
            and explicit_fingerprint != derived_fingerprint
        ):
            raise ValueError("ASR model fingerprint does not match local checkpoint")
        return str(resolved), derived_fingerprint
    return (
        _value(arguments.model, "ASR_MODEL", default=DEFAULT_ASR_MODEL),
        explicit_fingerprint,
    )


def _reuse_existing(
    *,
    output_root: Path,
    inventory: ASRV2Inventory,
    backend_config: WhisperASRConfig,
) -> ASRV2Summary | None:
    if not asr_v2_stage_is_complete(output_root):
        return None
    existing_inventory = ASRV2Inventory.model_validate_json(
        (output_root / "inventory.json").read_text(encoding="utf-8")
    )
    summary = ASRV2Summary.model_validate_json(
        (output_root / "summary.json").read_text(encoding="utf-8")
    )
    if existing_inventory.inventory_fingerprint != inventory.inventory_fingerprint:
        raise ValueError(
            "existing ASR V2 output uses different inputs; pass --overwrite"
        )
    expected_provenance = build_asr_v2_backend_provenance(
        runtime=backend_config.provenance(),
        baseline=inventory.baseline_asr_v1_backend_provenance,
    )
    if (
        summary.backend_provenance.configuration_fingerprint
        != expected_provenance.configuration_fingerprint
    ):
        raise ValueError(
            "existing ASR V2 output uses a different model/config; pass --overwrite"
        )
    return summary


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    audio_run_root = arguments.audio_run_root.expanduser().resolve(strict=True)
    output_root = asr_v2_output_root(audio_run_root, mode=arguments.mode)
    if arguments.regenerate_review:
        if arguments.dry_run or arguments.overwrite:
            raise ValueError(
                "--regenerate-review cannot be combined with --dry-run or --overwrite"
            )
        result = regenerate_asr_v2_review(
            output_root=output_root,
            expected_mode=arguments.mode,
        )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result

    inventory = build_asr_v2_inventory(
        audio_run_root=audio_run_root,
        mode=arguments.mode,
    )
    plan: dict[str, object] = {
        "mode": arguments.mode,
        "source_diarization_root": inventory.source_diarization_root,
        "source_diarization_inventory_fingerprint": (
            inventory.source_diarization_inventory_fingerprint
        ),
        "source_asr_v1_inventory_fingerprint": (
            inventory.source_asr_v1_inventory_fingerprint
        ),
        "source_target_count": inventory.source_target_count,
        "selected_target_count": inventory.selected_target_count,
        "selected_segment_count": inventory.selected_segment_count,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "output_root": str(output_root),
        "bounded_selection_applied": inventory.bounded_selection_applied,
        "parent_quota_applied": False,
        "donor_media_used": False,
        "cross_pair_jobs_created": 0,
        "production_inference_blocked": inventory.production_inference_blocked,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return plan
    if arguments.mode == "production":
        raise ValueError("production_blocked_pending_asr_v2_human_calibration")

    model_identifier, checkpoint_fingerprint = _model_identity(arguments)
    config = WhisperASRConfig(
        model_identifier=model_identifier,
        checkpoint_fingerprint=checkpoint_fingerprint,
        device=_value(arguments.device, "ASR_DEVICE", default=DEFAULT_ASR_DEVICE),
        compute_type=_value(
            arguments.compute_type,
            "ASR_COMPUTE_TYPE",
            default=DEFAULT_ASR_COMPUTE_TYPE,
        ),
    )
    build_asr_v2_backend_provenance(
        runtime=config.provenance(),
        baseline=inventory.baseline_asr_v1_backend_provenance,
    )
    if not arguments.overwrite:
        existing = _reuse_existing(
            output_root=output_root,
            inventory=inventory,
            backend_config=config,
        )
        if existing is not None:
            result = {
                **plan,
                "stage_status": "reused",
                "summary": existing.model_dump(mode="json"),
            }
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return result

    summary = run_asr_v2_transcription(
        inventory=inventory,
        output_root=output_root,
        backend=FasterWhisperASRBackend(config),
        overwrite=arguments.overwrite,
    )
    result = {
        **plan,
        "stage_status": "completed",
        "summary": summary.model_dump(mode="json"),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
