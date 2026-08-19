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

from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.target_audio_caption import (
    DEFAULT_DOTS3_CHECKPOINT_ID,
    DEFAULT_DOTS3_MODEL,
    Dots3TargetAudioCaptionConfig,
    OpenAIDots3TargetAudioCaptionBackend,
    build_target_audio_caption_inventory,
    run_target_audio_caption_pilot,
    target_audio_caption_output_root,
)


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
        description="Run the fixed 20-clip H3 target-audio-caption pilot",
    )
    parser.add_argument("--audio-run-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--checkpoint-id")
    parser.add_argument("--media-mode", choices=("file", "http"))
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--media-base-url")
    parser.add_argument("--timeout-seconds", type=_positive_float, default=600.0)
    parser.add_argument("--max-tokens", type=_positive_int, default=2048)
    return parser


def _value(
    explicit: str | None,
    environment_name: str,
    *,
    default: str | None = None,
) -> str:
    value = explicit or os.environ.get(environment_name) or default
    if value is None or not value.strip():
        raise ValueError(f"missing dots3/vLLM runtime input: {environment_name}")
    return value.strip()


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    audio_run_root = arguments.audio_run_root.expanduser().resolve(strict=True)
    inventory = build_target_audio_caption_inventory(audio_run_root=audio_run_root)
    output_root = target_audio_caption_output_root(audio_run_root)
    result: dict[str, object] = {
        "mode": "pilot20",
        "selected_target_count": inventory.selected_target_count,
        "selection_mode": inventory.selection_mode,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "output_root": str(output_root),
        "input_modality": "native_target_video_with_embedded_audio",
        "separate_audio_sent": False,
        "transcript_supplied": False,
        "final_renderer_applied": False,
    }
    if arguments.dry_run:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result

    media_mode = _value(
        arguments.media_mode,
        "DOTS3_MEDIA_MODE",
        default="file",
    )
    media_root = Path(
        _value(
            None if arguments.media_root is None else str(arguments.media_root),
            "DOTS3_MEDIA_ROOT",
        )
    )
    media_base_url = arguments.media_base_url or os.environ.get("DOTS3_MEDIA_BASE_URL")
    config = Dots3TargetAudioCaptionConfig(
        base_url=_value(arguments.base_url, "DOTS3_BASE_URL"),
        api_key=_value(arguments.api_key, "DOTS3_API_KEY", default="EMPTY"),
        served_model_name=_value(
            arguments.model,
            "DOTS3_MODEL",
            default=DEFAULT_DOTS3_MODEL,
        ),
        checkpoint_id=_value(
            arguments.checkpoint_id,
            "DOTS3_CHECKPOINT_ID",
            default=DEFAULT_DOTS3_CHECKPOINT_ID,
        ),
        media_resolver=MediaURLResolver(
            mode=media_mode,
            media_root=media_root,
            media_base_url=media_base_url if media_mode == "http" else None,
        ),
        timeout_seconds=arguments.timeout_seconds,
        max_tokens=arguments.max_tokens,
    )
    summary = run_target_audio_caption_pilot(
        inventory=inventory,
        output_root=output_root,
        backend=OpenAIDots3TargetAudioCaptionBackend(config),
        overwrite=arguments.overwrite,
    )
    result["summary"] = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
