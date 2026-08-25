#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.qwen3_asr_review import (
    FFmpegQwen3ASRReviewMediaBackend,
    generate_qwen3_asr_review,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a model-free video review for existing Qwen3 ASR rows",
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    output_root = arguments.output_root or (
        arguments.audio_production_root / "asr_review"
    )
    manifest = generate_qwen3_asr_review(
        audio_production_root=arguments.audio_production_root,
        output_root=output_root,
        media_backend=FFmpegQwen3ASRReviewMediaBackend(
            ffmpeg=arguments.ffmpeg,
            timeout_seconds=arguments.timeout_seconds,
        ),
        overwrite=arguments.overwrite,
    )
    result = {
        "audio_production_root": str(
            arguments.audio_production_root.expanduser().resolve(strict=True)
        ),
        "review_output_root": str(output_root.expanduser().resolve(strict=True)),
        "model_loaded": False,
        "model_calls": 0,
        "manifest": manifest.model_dump(mode="json"),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
