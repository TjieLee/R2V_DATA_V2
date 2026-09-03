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

from r2v_data_v2.h3.qwen_speech_presentation_recaption import (
    OpenAIQwenPresentationBackend,
    QwenPresentationConfig,
    run_qwen_speech_presentation_recaption,
)
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the model-neutral Qwen speech-presentation H3 shadow",
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_PRESENTATION_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("QWEN_PRESENTATION_API_KEY", "EMPTY")
    )
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--media-mode", choices=("file", "http"), default="file")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--media-base-url")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    resolver = MediaURLResolver(
        mode=arguments.media_mode,
        media_root=arguments.media_root,
        media_base_url=arguments.media_base_url,
    )
    backend = OpenAIQwenPresentationBackend(
        QwenPresentationConfig(
            base_url=arguments.base_url,
            media_resolver=resolver,
            api_key=arguments.api_key,
            served_model_name=arguments.served_model_name,
            checkpoint_id=arguments.checkpoint_id,
            timeout_seconds=arguments.timeout_seconds,
            max_tokens=arguments.max_tokens,
            temperature=arguments.temperature,
            top_p=arguments.top_p,
            top_k=arguments.top_k,
            min_p=arguments.min_p,
            presence_penalty=arguments.presence_penalty,
            repetition_penalty=arguments.repetition_penalty,
        )
    )
    summary = run_qwen_speech_presentation_recaption(
        audio_production_root=arguments.audio_production_root,
        case_manifest_path=arguments.case_manifest,
        output_root=arguments.output_root,
        backend=backend,
        overwrite=arguments.overwrite,
    )
    result = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
