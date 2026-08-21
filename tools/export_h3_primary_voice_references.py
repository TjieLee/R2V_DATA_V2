from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.primary_voice import export_primary_voice_references


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the calibrated V1 turn-quality gate and export primary voice "
            "references from an existing Audio Binding pilot without rerunning models"
        ),
    )
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    summary = export_primary_voice_references(
        pilot_root=arguments.pilot_root,
        output_root=arguments.output_root,
        audio_backend=FFmpegAudioMediaBackend(
            ffmpeg=arguments.ffmpeg,
            ffprobe=arguments.ffprobe,
        ),
        overwrite=arguments.overwrite,
    )
    result = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
