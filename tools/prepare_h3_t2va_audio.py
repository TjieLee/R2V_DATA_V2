"""Select JEA shots and bootstrap a new no-reference Audio workspace (CPU only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3.audio_backends import FFmpegAudioMediaBackend
from r2v_data_v2.h3.t2va_source import prepare_t2va_audio, select_t2va_shots


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot-manifest", type=Path, required=True)
    parser.add_argument("--clips-root", type=Path)
    parser.add_argument("--source-videos-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case-manifest", type=Path)
    selection.add_argument("--sample-size", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument(
        "--shot-index-root",
        type=Path,
        help="Writable JSONL offset cache (outside source directory)",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    selected = select_t2va_shots(
        args.shot_manifest,
        clips_root=args.clips_root,
        source_videos_root=args.source_videos_root,
        case_manifest=args.case_manifest,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        shot_index_root=args.shot_index_root,
    )
    if not args.dry_run:
        prepare_t2va_audio(
            selected,
            output_root=args.output_root,
            audio_backend=FFmpegAudioMediaBackend(
                ffmpeg=args.ffmpeg, ffprobe=args.ffprobe
            ),
        )
    return {
        "clip_uids": [s.clip_uid for s in selected.shots],
        "shot_manifest_sha256": selected.shot_manifest_sha256,
        "output_root": str(args.output_root.resolve()),
        "model_call_count": 0,
        "dry_run": args.dry_run,
    }


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
