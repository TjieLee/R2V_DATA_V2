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

from r2v_data_v2.h3.qwen3_asr import (
    QWEN3_ASR_MODEL_IDENTIFIER,
    Qwen3ASRBackend,
    Qwen3ASRConfiguration,
    run_qwen3_asr,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe exact DiariZen segments with local Qwen3-ASR-1.7B",
    )
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--visual-runs-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    environment = os.environ.get("QWEN3_ASR_ENV")
    if not environment:
        raise ValueError("QWEN3_ASR_ENV must identify the isolated qwen-asr==0.0.6 env")
    configuration = Qwen3ASRConfiguration.from_environment()
    visual_root = arguments.visual_production_root.expanduser().resolve(strict=True)
    visual_runs_root = arguments.visual_runs_root.expanduser().resolve(strict=True)
    audio_root = arguments.audio_production_root.expanduser().resolve(strict=True)
    summary = run_qwen3_asr(
        diarization_root=audio_root / "diarization",
        source_visual_production_root=str(visual_root),
        output_root=audio_root / "asr",
        backend=Qwen3ASRBackend(configuration),
        ffmpeg=arguments.ffmpeg,
        overwrite=arguments.overwrite,
    )
    result = {
        "audio_production_root": str(audio_root),
        "visual_production_root": str(visual_root),
        "visual_runs_root": str(visual_runs_root),
        "asr_output_root": str(audio_root / "asr"),
        "selected_asr_model": QWEN3_ASR_MODEL_IDENTIFIER,
        "qwen3_asr_env": environment,
        "summary": summary.model_dump(mode="json"),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
