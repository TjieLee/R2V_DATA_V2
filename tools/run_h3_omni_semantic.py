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

from r2v_data_v2.h3.semantic_augmentation import (
    DEFAULT_MAX_BASE64_BYTES,
    DEFAULT_QWEN_OMNI_MODEL,
    OpenAIQwenOmniBackend,
    QwenOmniSemanticConfig,
    SemanticInventory,
    SemanticProductionSummary,
    build_semantic_inventory,
    run_semantic_augmentation,
    semantic_output_root,
    semantic_stage_is_complete,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment frozen H3 production target clips with Qwen Omni semantics",
    )
    parser.add_argument("--audio-run-root", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("pilot20", "production"),
        required=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument(
        "--max-base64-bytes",
        type=_positive_integer,
        default=None,
    )
    parser.add_argument("--timeout-seconds", type=_positive_float, default=600.0)
    parser.add_argument("--max-tokens", type=_positive_integer, default=4096)
    return parser


def _environment_or_argument(
    explicit: str | None,
    environment_name: str,
    *,
    default: str | None = None,
) -> str:
    value = explicit or os.environ.get(environment_name) or default
    if value is None or not value.strip():
        raise ValueError(f"missing Qwen Omni runtime input: {environment_name}")
    return value.strip()


def _maximum_base64_bytes(explicit: int | None) -> int:
    if explicit is not None:
        return explicit
    raw = os.environ.get("QWEN_OMNI_MAX_BASE64_BYTES")
    if raw is None:
        return DEFAULT_MAX_BASE64_BYTES
    return _positive_integer(raw)


def _reuse_existing(
    *,
    output_root: Path,
    inventory: SemanticInventory,
    model_identifier: str,
) -> SemanticProductionSummary | None:
    if not semantic_stage_is_complete(output_root):
        return None
    existing_inventory = SemanticInventory.model_validate_json(
        (output_root / "inventory.json").read_text(encoding="utf-8")
    )
    summary = SemanticProductionSummary.model_validate_json(
        (output_root / "summary.json").read_text(encoding="utf-8")
    )
    if existing_inventory.inventory_fingerprint != inventory.inventory_fingerprint:
        raise ValueError(
            "existing semantic output uses different target inputs; pass --overwrite"
        )
    if summary.model_identifier != model_identifier:
        raise ValueError(
            "existing semantic output uses a different model; pass --overwrite"
        )
    return summary


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    audio_run_root = arguments.audio_run_root.expanduser().resolve(strict=True)
    pairs_root = (audio_run_root / "production" / "pairs").resolve(strict=True)
    inventory = build_semantic_inventory(
        pairs_root=pairs_root,
        mode=arguments.mode,
    )
    output_root = semantic_output_root(audio_run_root, mode=arguments.mode)
    model_identifier = _environment_or_argument(
        arguments.model,
        "QWEN_OMNI_MODEL",
        default=DEFAULT_QWEN_OMNI_MODEL,
    )
    plan = {
        "mode": arguments.mode,
        "source_pairs_path": inventory.source_pairs_path,
        "source_target_count": inventory.source_target_count,
        "selected_target_count": inventory.selected_target_count,
        "multi_subject_target_count": sum(
            item.subject_count > 1 for item in inventory.jobs
        ),
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "output_root": str(output_root),
        "model_identifier": model_identifier,
        "bounded_selection_applied": inventory.bounded_selection_applied,
        "parent_quota_applied": False,
        "donor_media_used": False,
    }
    if arguments.dry_run:
        print(json.dumps(plan, ensure_ascii=False, sort_keys=True))
        return plan

    if not arguments.overwrite:
        existing = _reuse_existing(
            output_root=output_root,
            inventory=inventory,
            model_identifier=model_identifier,
        )
        if existing is not None:
            result = {
                **plan,
                "stage_status": "reused",
                "summary": existing.model_dump(mode="json"),
            }
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return result

    config = QwenOmniSemanticConfig(
        base_url=_environment_or_argument(
            arguments.base_url,
            "QWEN_OMNI_BASE_URL",
        ),
        api_key=_environment_or_argument(
            arguments.api_key,
            "DASHSCOPE_API_KEY",
        ),
        model=model_identifier,
        max_base64_bytes=_maximum_base64_bytes(arguments.max_base64_bytes),
        timeout_seconds=arguments.timeout_seconds,
        max_tokens=arguments.max_tokens,
    )
    summary = run_semantic_augmentation(
        inventory=inventory,
        output_root=output_root,
        backend=OpenAIQwenOmniBackend(config),
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
