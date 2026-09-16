"""Small visual count/rational-FPS contract, independent of any model backend."""

import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class VideoTimeline:
    frame_count: int
    fps: float
    fps_num: int | None
    fps_den: int | None
    width: int
    height: int
    duration_seconds: float

    @property
    def rate(self) -> Fraction:
        if self.fps_num is not None and self.fps_den is not None:
            return Fraction(self.fps_num, self.fps_den)
        return Fraction(str(self.fps)).limit_denominator(1_000_000)


def inspect_video_timeline(path: Path) -> VideoTimeline:
    # No reusable neutral ffprobe helper exists; H3's helpers are audio-specific.
    # nb_read_frames counts decoded frames, unlike the container's nb_frames tag.
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_frames", "-show_entries",
        "stream=nb_read_frames,avg_frame_rate,r_frame_rate,width,height,time_base:frame=best_effort_timestamp",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload["streams"]
    if len(streams) != 1:
        raise ValueError(f"Expected one selected video stream: {path}")
    stream = streams[0]
    count = int(stream["nb_read_frames"])
    try:
        rate = Fraction(stream["avg_frame_rate"])
    except (ValueError, ZeroDivisionError, KeyError):
        rate = Fraction(stream["r_frame_rate"])
    width, height = int(stream["width"]), int(stream["height"])
    if rate <= 0 or count < 1 or min(width, height) < 1:
        raise ValueError(f"Invalid decoded video timeline: {path}")
    # Bernini's FPS-based sampler cannot preserve arbitrary VFR time positions.
    # Check decoded presentation timestamps, not just an average container rate.
    time_base = Fraction(stream["time_base"])
    pts = [int(frame["best_effort_timestamp"]) * time_base for frame in payload["frames"]]
    if time_base <= 0 or len(pts) != count:
        raise ValueError(f"Incomplete decoded timestamp evidence: {path}")
    tolerance = min(time_base, Fraction(1, 10) / rate)
    if any(pts[i] <= pts[i-1] or abs(pts[i] - pts[0] - i / rate) > tolerance
           for i in range(1, count)):
        raise ValueError(f"Unsupported variable/invalid frame cadence: {path}")
    duration = float(pts[-1] - pts[0] + 1 / rate)
    return VideoTimeline(count, float(rate), rate.numerator, rate.denominator,
                         width, height, duration)


def validate_matching_timeline(source: VideoTimeline, output: VideoTimeline) -> None:
    for value in (source, output):
        if (value.frame_count < 1 or min(value.width, value.height) < 1
                or not math.isfinite(value.fps) or value.fps <= 0
                or not math.isfinite(value.duration_seconds) or value.duration_seconds <= 0
                or not math.isclose(float(value.rate), value.fps, rel_tol=1e-8)):
            raise ValueError("Invalid timeline")
    if output.frame_count != source.frame_count:
        raise ValueError(f"Frame count mismatch: {output.frame_count} != {source.frame_count}")
    if not math.isclose(output.fps, source.fps, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"FPS mismatch: {output.fps} != {source.fps}")
    if abs(output.duration_seconds - source.duration_seconds) > 1 / source.fps + 1e-9:
        raise ValueError("Duration mismatch exceeds one source frame")


def transcode_timeline(source: Path, output: Path, *, rate: Fraction, prefix_filter: str, lossless=False):
    """Only frame-index trim/tail clone + explicit timestamps; never an FPS filter."""
    if output.exists() or source.resolve() == output.resolve():
        raise FileExistsError(f"Refusing to overwrite timeline output: {output}")
    num, den = rate.numerator, rate.denominator
    filters = f"{prefix_filter}," if prefix_filter else ""
    filters += f"settb=expr=1/{num},setpts=N*{den}"
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source),
        "-map", "0:v:0", "-an", "-vf", filters,
        "-r", f"{num}/{den}", "-fps_mode", "cfr",
        "-enc_time_base", f"{den}:{num}", "-video_track_timescale", str(num),
        "-c:v", "libx264", "-preset", "slow", "-crf", "0" if lossless else "12",
        "-pix_fmt", "yuv420p", str(output),
    ]
    with (output.parent / "timeline.log").open("a") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
