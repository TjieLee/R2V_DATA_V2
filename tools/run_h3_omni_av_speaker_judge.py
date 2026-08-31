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

from r2v_data_v2.h3.omni_av_speaker_judge import (
    DEFAULT_MODEL,
    OmniAVSpeakerJudgeConfig,
    OpenAIOmniAVSpeakerJudge,
    SubprocessOmniAVSpeakerMediaBackend,
    run_omni_av_speaker_judge_pilot,
)
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the read-only Qwen3-Omni AV speaker judge pilot",
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN3_OMNI_BASE_URL", "http://127.0.0.1:8091/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("QWEN3_OMNI_API_KEY", "EMPTY"))
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-id", default=DEFAULT_MODEL)
    parser.add_argument("--media-mode", choices=("file", "http"), default="file")
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--media-base-url")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    resolver = MediaURLResolver(
        mode=arguments.media_mode,
        media_root=arguments.media_root,
        media_base_url=arguments.media_base_url,
    )
    backend = OpenAIOmniAVSpeakerJudge(
        OmniAVSpeakerJudgeConfig(
            base_url=arguments.base_url,
            media_resolver=resolver,
            api_key=arguments.api_key,
            served_model_name=arguments.served_model_name,
            checkpoint_id=arguments.checkpoint_id,
            timeout_seconds=arguments.timeout_seconds,
            max_tokens=arguments.max_tokens,
        )
    )
    media_backend = SubprocessOmniAVSpeakerMediaBackend(
        python_path=arguments.python,
        ffmpeg=arguments.ffmpeg,
        timeout_seconds=arguments.timeout_seconds,
    )
    summary = run_omni_av_speaker_judge_pilot(
        audio_production_root=arguments.audio_production_root,
        case_manifest_path=arguments.case_manifest,
        output_root=arguments.output_root,
        backend=backend,
        media_backend=media_backend,
        overwrite=arguments.overwrite,
    )
    result = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
