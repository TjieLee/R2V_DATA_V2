#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.jea_audio_production import (
    jea_production_paths,
    materialize_canonical_audio_clips,
)
from r2v_data_v2.h3.visual_production_source import (
    VisualProductionClip,
    load_visual_production_inventory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill model-free canonical H3 full audio and manifest",
    )
    parser.add_argument("--visual-production-root", type=Path, required=True)
    parser.add_argument("--visual-runs-root", type=Path, required=True)
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def _video_duration(item: VisualProductionClip, *, ffprobe: str) -> float:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "json",
            item.sample.target_video,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("canonical target video must expose one audio stream")
    return float(streams[0]["duration"])


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    visual = load_visual_production_inventory(
        visual_production_root=arguments.visual_production_root,
        visual_runs_root=arguments.visual_runs_root,
    )
    production_root = arguments.audio_production_root.expanduser().resolve(strict=True)
    audio_root = jea_production_paths(production_root).audio
    summary = materialize_canonical_audio_clips(
        visual_inventory=visual,
        audio_root=audio_root,
        audio_backend=FFmpegAudioMediaBackend(
            ffmpeg=arguments.ffmpeg,
            ffprobe=arguments.ffprobe,
        ),
        duration_resolver=lambda item: _video_duration(
            item,
            ffprobe=arguments.ffprobe,
        ),
    )
    result = {
        "model_calls": 0,
        "visual_canonical_clip_count": summary.visual_canonical_clip_count,
        "canonical_audio_clip_count": summary.canonical_audio_clip_count,
        "manifest_path": str(audio_root / "canonical_clips.jsonl"),
        "summary_path": str(audio_root / "canonical_clips_summary.json"),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
