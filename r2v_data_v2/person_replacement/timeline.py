"""Backend-neutral decoded timeline contract; no model dependencies."""

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoTimeline:
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float


def inspect_video_timeline(path: Path) -> VideoTimeline:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unreadable video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid source fps: {path}")
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        count, size = 0, None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if size is not None and size != (width, height):
                raise ValueError("Variable resolution video is unsupported")
            size = (width, height)
            count += 1
        if not count or (reported_count > 0 and reported_count != count):
            raise ValueError(f"Incomplete decoded frame count: {count}/{reported_count}: {path}")
        return VideoTimeline(count, fps, *size, count / fps)
    finally:
        capture.release()


def validate_matching_timeline(source: VideoTimeline, output: VideoTimeline) -> None:
    for timeline in (source, output):
        if (timeline.frame_count <= 0 or timeline.width <= 0 or timeline.height <= 0
                or not math.isfinite(timeline.fps) or timeline.fps <= 0
                or not math.isfinite(timeline.duration_seconds) or timeline.duration_seconds <= 0):
            raise ValueError("Invalid video timeline")
    if source.frame_count != output.frame_count:
        raise ValueError(f"Output frame count {output.frame_count} != source {source.frame_count}")
    if not math.isclose(source.fps, output.fps, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"Output fps {output.fps} != source {source.fps}")
    if abs(source.duration_seconds - output.duration_seconds) > 1 / source.fps + 1e-9:
        raise ValueError("Output duration differs by more than one source frame")
