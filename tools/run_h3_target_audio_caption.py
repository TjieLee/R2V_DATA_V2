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

from r2v_data_v2.h3.jea_target_audio_caption import (
    DEFAULT_DOTS3_CHECKPOINT_ID,
    DEFAULT_DOTS3_MODEL,
    DEFAULT_QWEN3_OMNI_CHECKPOINT_ID,
    DEFAULT_QWEN3_OMNI_MODEL,
    JEATargetAudioCaptionConfig,
    OpenAIJEATargetAudioCaptionBackend,
    build_jea_target_audio_caption_inventory,
    run_jea_target_audio_caption,
    target_audio_caption_output_root,
)
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run JEA target-audio-caption with a selectable vLLM backend",
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("dots3", "qwen3-omni"), required=True)
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="also send target video to Qwen3-Omni; invalid for Dots3",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--media-mode", choices=("file", "http"))
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--media-base-url")
    parser.add_argument("--timeout-seconds", type=_positive_float, default=600.0)
    parser.add_argument("--max-tokens", type=_positive_int, default=2048)
    parser.add_argument("--max-concurrency", type=_positive_int, default=1)
    return parser


def _value(
    explicit: str | None,
    environment_name: str,
    *,
    default: str | None = None,
) -> str:
    value = explicit or os.environ.get(environment_name) or default
    if value is None or not value.strip():
        raise ValueError(f"missing vLLM runtime input: {environment_name}")
    return value.strip()


def _backend_config(arguments: argparse.Namespace) -> JEATargetAudioCaptionConfig:
    if arguments.backend == "dots3":
        family = "dots3"
        prefix = "DOTS3"
        default_model = DEFAULT_DOTS3_MODEL
        default_checkpoint = DEFAULT_DOTS3_CHECKPOINT_ID
    else:
        family = "qwen3_omni"
        prefix = "QWEN3_OMNI"
        default_model = DEFAULT_QWEN3_OMNI_MODEL
        default_checkpoint = DEFAULT_QWEN3_OMNI_CHECKPOINT_ID
    media_mode = _value(
        arguments.media_mode,
        f"{prefix}_MEDIA_MODE",
        default="file",
    )
    media_root = Path(
        _value(
            None if arguments.media_root is None else str(arguments.media_root),
            f"{prefix}_MEDIA_ROOT",
        )
    )
    media_base_url = arguments.media_base_url or os.environ.get(
        f"{prefix}_MEDIA_BASE_URL"
    )
    return JEATargetAudioCaptionConfig(
        backend_family=family,
        base_url=_value(arguments.base_url, f"{prefix}_BASE_URL"),
        api_key=_value(arguments.api_key, f"{prefix}_API_KEY", default="EMPTY"),
        served_model_name=_value(
            arguments.model,
            f"{prefix}_MODEL",
            default=default_model,
        ),
        checkpoint_id=_value(
            arguments.checkpoint_id,
            f"{prefix}_CHECKPOINT_ID",
            default=default_checkpoint,
        ),
        media_resolver=MediaURLResolver(
            mode=media_mode,
            media_root=media_root,
            media_base_url=media_base_url if media_mode == "http" else None,
        ),
        include_video=arguments.include_video,
        timeout_seconds=arguments.timeout_seconds,
        max_tokens=arguments.max_tokens,
    )


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.backend == "dots3" and arguments.include_video:
        parser.error("--include-video is only valid with --backend qwen3-omni")
    production_root = arguments.audio_production_root.expanduser().resolve(strict=True)
    backend_family = "dots3" if arguments.backend == "dots3" else "qwen3_omni"
    inventory = build_jea_target_audio_caption_inventory(
        audio_production_root=production_root,
    )
    output_root = arguments.output_root or target_audio_caption_output_root(
        production_root,
        backend_family=backend_family,
    )
    result: dict[str, object] = {
        "backend_family": backend_family,
        "target_clip_count": inventory.target_clip_count,
        "readable_segment_count": inventory.readable_segment_count,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "output_root": str(output_root.expanduser().resolve(strict=False)),
        "input_modality": (
            "native_target_video_with_embedded_audio"
            if backend_family == "dots3"
            else (
                "target_video_plus_canonical_full_audio"
                if arguments.include_video
                else "canonical_full_audio_only"
            )
        ),
        "transcript_supplied": False,
        "entity_id_supplied": False,
        "donor_media_used": False,
        "max_concurrency": arguments.max_concurrency,
        "model_calls": 0,
    }
    if arguments.dry_run:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result

    backend = OpenAIJEATargetAudioCaptionBackend(_backend_config(arguments))
    summary = run_jea_target_audio_caption(
        inventory=inventory,
        output_root=output_root,
        backend=backend,
        overwrite=arguments.overwrite,
        max_concurrency=arguments.max_concurrency,
    )
    result["model_calls"] = (
        summary.initial_call_count
        + summary.repair_call_count
        + summary.semantic_fallback_initial_call_count
        + summary.semantic_fallback_repair_call_count
        + summary.overall_audio_description_initial_call_count
        + summary.overall_audio_description_fallback_initial_call_count
        + summary.overall_audio_description_repair_call_count
    )
    result["summary"] = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
