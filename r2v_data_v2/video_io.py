from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from r2v_data_v2.config import FramesConfig, PipelineConfig
from r2v_data_v2.manifest import iter_source_records


@dataclass(frozen=True)
class FrameSamplingStats:
    processed: int = 0
    skipped_existing: int = 0
    failed: int = 0


def sample_frame_indices(total_frames: int, count: int = 10) -> list[int]:
    if total_frames < 1:
        raise ValueError("video must contain at least one frame")
    if count < 1:
        raise ValueError("sample count must be positive")
    return [
        min(total_frames - 1, max(0, round((index + 0.5) * total_frames / count - 0.5)))
        for index in range(count)
    ]


def resize_without_upscaling(frame: np.ndarray, max_side: int) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("video frame must have HWC BGR shape")
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    stored_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, stored_size, interpolation=cv2.INTER_AREA)


def _write_jpeg(path: Path, frame: np.ndarray, quality: int) -> None:
    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise RuntimeError(f"failed to encode JPEG: {path}")
    path.write_bytes(encoded.tobytes())


def sample_video_frames(
    *,
    clip_uid: str,
    video_path: str | Path,
    output_dir: str | Path,
    config: FramesConfig,
) -> dict[str, object]:
    video = Path(video_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video}")
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        indices = sample_frame_indices(total_frames, config.count)
        stored_width = stored_height = 0
        for slot, source_index in enumerate(indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
            success, frame = capture.read()
            if not success or frame is None:
                raise RuntimeError(
                    f"failed to decode frame {source_index} from video: {video}"
                )
            stored = resize_without_upscaling(frame, config.max_side)
            stored_height, stored_width = stored.shape[:2]
            _write_jpeg(
                destination / f"frame_{slot:02d}.jpg",
                stored,
                config.jpeg_quality,
            )
    finally:
        capture.release()

    metadata: dict[str, object] = {
        "clip_uid": clip_uid,
        "video_path": str(video),
        "source_frame_count": total_frames,
        "source_fps": fps,
        "sampled_indices": indices,
        "sampled_timestamps": [
            (source_index / fps if fps > 0 else None) for source_index in indices
        ],
        "original_size": [width, height],
        "stored_size": [stored_width, stored_height],
    }
    (destination / "frames.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def _append_failure(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def sample_manifest_frames(
    config: PipelineConfig,
    *,
    overwrite: bool = False,
) -> FrameSamplingStats:
    output_root = config.ensure_output_root()
    manifest_path = output_root / "manifests" / "source.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError("run Stage 00 before frame sampling")
    processed = skipped = failed = 0
    for record in iter_source_records(manifest_path):
        clip = str(record["clip_uid"])
        destination = output_root / "frames" / clip
        if (destination / "frames.json").is_file() and not overwrite:
            skipped += 1
            continue
        if overwrite and destination.is_dir():
            for path in destination.glob("frame_*.jpg"):
                path.unlink()
            (destination / "frames.json").unlink(missing_ok=True)
        try:
            sample_video_frames(
                clip_uid=clip,
                video_path=str(record["video_path"]),
                output_dir=destination,
                config=config.frames,
            )
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the batch
            _append_failure(
                output_root / "logs" / "frame_failed.jsonl",
                {
                    "clip_uid": clip,
                    "video_path": record["video_path"],
                    "error": str(exc),
                },
            )
            failed += 1
            continue
        processed += 1
    return FrameSamplingStats(processed, skipped, failed)


def stats_dict(stats: FrameSamplingStats) -> dict[str, int]:
    return asdict(stats)
