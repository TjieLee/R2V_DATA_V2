#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path


def _visible_samples(
    samples: list[dict[str, object]],
    timestamp: float,
) -> list[dict[str, object]]:
    by_track: dict[str, list[dict[str, object]]] = {}
    for sample in samples:
        by_track.setdefault(str(sample["face_track_id"]), []).append(sample)
    visible: list[dict[str, object]] = []
    for track_id, track_samples in sorted(by_track.items()):
        nearest = min(
            track_samples,
            key=lambda item: (
                abs(float(item["timestamp_seconds"]) - timestamp),
                int(item["frame_index"]),
            ),
        )
        if abs(float(nearest["timestamp_seconds"]) - timestamp) <= 0.06:
            visible.append(nearest)
    return visible


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--window-start", type=float, required=True)
    parser.add_argument("--window-end", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    arguments = parser.parse_args(argv)
    if (
        not math.isfinite(arguments.window_start)
        or not math.isfinite(arguments.window_end)
        or arguments.window_start < 0
        or arguments.window_end <= arguments.window_start
    ):
        raise ValueError("neutral review window is invalid")

    import cv2

    source = arguments.source.expanduser().resolve(strict=True)
    timeline = json.loads(arguments.timeline.read_text(encoding="utf-8"))
    if not isinstance(timeline, dict) or not isinstance(timeline.get("samples"), list):
        raise TypeError("neutral face timeline is invalid")
    samples = timeline["samples"]
    source_width = int(timeline["source_width"])
    source_height = int(timeline["source_height"])
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open neutral review source: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("neutral review video metadata is invalid")
    start_frame = math.floor(arguments.window_start * fps)
    end_frame = math.ceil(arguments.window_end * fps)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    output = arguments.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    pretrim = output.with_name(f".{output.stem}.pretrim.mp4")
    writer = cv2.VideoWriter(
        str(pretrim),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError("could not initialize neutral review video writer")
    frame_index = start_frame
    try:
        while frame_index < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            for sample in _visible_samples(samples, timestamp):
                x1, y1, x2, y2 = [float(value) for value in sample["bbox_xyxy"]]
                scaled = (
                    round(x1 * width / source_width),
                    round(y1 * height / source_height),
                    round(x2 * width / source_width),
                    round(y2 * height / source_height),
                )
                sx1, sy1, sx2, sy2 = scaled
                color = (220, 220, 220)
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
                cv2.putText(
                    frame,
                    str(sample["label"]),
                    (max(0, sx1), max(20, sy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        capture.release()
    offset = arguments.window_start - start_frame / fps
    try:
        completed = subprocess.run(
            [
                arguments.ffmpeg,
                "-y",
                "-ss",
                f"{offset:.9f}",
                "-i",
                str(pretrim),
                "-t",
                f"{arguments.window_end - arguments.window_start:.9f}",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output),
                "-loglevel",
                "error",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        pretrim.unlink(missing_ok=True)
    if (
        completed.returncode != 0
        or not output.is_file()
        or output.stat().st_size == 0
    ):
        raise RuntimeError(
            "neutral review video trim failed: "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
