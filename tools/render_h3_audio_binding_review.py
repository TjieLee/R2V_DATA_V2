from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _review_mux_command(
    *,
    ffmpeg: str,
    silent_video: Path,
    source_video: Path,
    output: Path,
) -> list[str]:
    return [
        ffmpeg,
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-ac:a",
        "2",
        "-shortest",
        str(output),
        "-loglevel",
        "error",
    ]


def _state_at(bindings: list[dict[str, object]], timestamp: float) -> str:
    for binding in bindings:
        if float(binding["start_time"]) <= timestamp < float(binding["end_time"]):
            return str(binding["status"])
    return "ambiguous"


def _visible_samples(
    samples: list[dict[str, object]],
    timestamp: float,
) -> list[dict[str, object]]:
    by_track: dict[str, list[dict[str, object]]] = {}
    for item in samples:
        by_track.setdefault(str(item["face_track_id"]), []).append(item)
    visible = []
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
    parser.add_argument("--source", required=True)
    parser.add_argument("--timeline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args(argv)

    import cv2

    source = Path(args.source).expanduser().resolve(strict=True)
    timeline = json.loads(Path(args.timeline).read_text(encoding="utf-8"))
    samples = timeline["samples"]
    bindings = timeline["bindings"]
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"could not open review source: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("review source video metadata is invalid")
    output = Path(args.output).expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    silent = output.with_name(f".{output.stem}.silent.mp4")
    writer = cv2.VideoWriter(
        str(silent),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError("could not initialize review visualization writer")
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            current_state = _state_at(bindings, timestamp)
            visible = _visible_samples(samples, timestamp)
            for item in sorted(visible, key=lambda value: str(value["face_track_id"])):
                x1, y1, x2, y2 = [round(value) for value in item["bbox_xyxy"]]
                active = bool(item["backend_native_active"])
                color = (0, 210, 0) if active else (0, 0, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = (
                    f"{item['face_track_id']} entity={item['entity_id']} "
                    f"score={float(item['raw_class1_logit']):.2f} "
                    f"{'active' if active else 'silent'} "
                    f"state={item['binding_status']}"
                )
                cv2.putText(
                    frame,
                    label,
                    (max(0, x1), max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            cv2.putText(
                frame,
                f"state={current_state}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
            frame_index += 1
    finally:
        writer.release()
        capture.release()

    command = _review_mux_command(
        ffmpeg=args.ffmpeg,
        silent_video=silent,
        source_video=source,
        output=output,
    )
    result = subprocess.run(command, check=False)
    silent.unlink(missing_ok=True)
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError("ffmpeg failed to mux review visualization")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
