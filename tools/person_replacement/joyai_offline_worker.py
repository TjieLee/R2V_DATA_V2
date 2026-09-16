"""Thin offline bridge to pinned JoyAI; run only in its external environment."""

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from r2v_data_v2.person_replacement.timeline import (
    VideoTimeline,
    inspect_video_timeline,
    validate_matching_timeline,
)


def decode_frames(path):
    import cv2
    from PIL import Image

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Unreadable source: {path}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


@contextmanager
def video_writer(path, fps, width, height):
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        if not writer.isOpened():
            raise ValueError(f"Cannot open output encoder: {path}")

        def write(frame):
            rgb = np.asarray(frame)
            if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
                raise ValueError("Unexpected JoyAI output frame shape/dtype")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        yield write
    finally:
        writer.release()


def stream_frames(runtime, settings, prompt, frames, frame_count, fps, write):
    """Drain each submitted chunk: bounded memory, sequence-checked real frames only."""
    session = runtime.create_v2v_session(prompt, settings=settings, ref_image=None)
    fed = written = 0

    def consume(result):
        nonlocal written
        pixels = result.jpegs
        metas = result.source_metas
        if pixels is None or len(pixels) != len(metas):
            raise ValueError("JoyAI output/metadata length mismatch")
        valid = len(pixels) if result.valid_count is None else result.valid_count
        if not 0 < valid <= len(pixels):
            raise ValueError("Invalid JoyAI padding valid_count")
        for frame, meta in zip(pixels[:valid], metas[:valid], strict=True):
            if written >= frame_count or meta.get("seq") != written + 1:
                raise ValueError("JoyAI output frame ordering/count mismatch")
            write(frame)
            written += 1

    def drain():
        deadline = time.monotonic() + 600
        while session.inflight_chunks():
            result = session.wait_async_result(timeout=0.1)
            if result is not None:
                consume(result)
                deadline = time.monotonic() + 600
            elif time.monotonic() >= deadline:
                raise TimeoutError("JoyAI produced no chunk for 600 seconds")

    try:
        for frame in frames:
            fed += 1
            if fed > frame_count:
                raise ValueError("Source grew after timeline inspection")
            for result in session.push_frame(
                frame, frame_meta={"seq": fed, "t_capture_ms": (fed - 1) * 1000 / fps},
            ):
                consume(result)
            drain()
        if fed != frame_count:
            raise ValueError(f"Source decode incomplete: {fed}/{frame_count}")
        session.flush_pending()  # Official tail repeat-padding; result carries valid_count.
        drain()
        if written != frame_count:
            raise ValueError(f"JoyAI output frame count {written} != source {frame_count}")
    finally:
        session.close()


def choose_bucket(width, height, buckets, alignment):
    sizes = {(b[3], b[4]) for b in buckets if b[3] % alignment == b[4] % alignment == 0}
    # Preserve orientation; choose the closest official aspect ratio, never crop.
    sizes = {(h, w) for h, w in sizes if (w > h) == (width > height)
             and (w < h) == (width < height)}
    if not sizes:
        raise ValueError("No official JoyAI bucket supports this orientation/alignment")
    return min(sizes, key=lambda hw: (abs(math.log((hw[1] / hw[0]) / (width / height))), hw))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text())
    source, output = Path(request["source"]), Path(request["output"])
    if output.exists() or output == source or output.parent != args.request.resolve().parent:
        raise ValueError("Worker requires a new case-local output")
    expected = VideoTimeline(**request["timeline"])
    actual = inspect_video_timeline(source)
    validate_matching_timeline(expected, actual)
    if (expected.width, expected.height) != (actual.width, actual.height):
        raise ValueError("Source resolution changed after inspection")

    # Imports below happen only in the subprocess, never in parent Qwen/dry-run.
    sys.path.insert(0, str(Path(request["joyai_code_root"]) / "deploy"))
    from xvideo.config import generate_video_image_bucket
    from xvideo.serving.joyomni_streaming import JoyOmniRuntime, StreamingSettings

    runtime = JoyOmniRuntime.load(
        request["joyai_dit_ckpt"], vae_ckpt=request["joyai_vae_ckpt"],
        text_encoder_ckpt=request["joyai_text_encoder_ckpt"], seed=request["seed"],
    )
    buckets = generate_video_image_bucket(
        bs_img=0, bs_vid=1, bs_mimg=0, bs_mvid=0, min_temporal=1, max_temporal=1,
        vid_basesizes=[(720, 1248)],
    )
    height, width = choose_bucket(actual.width, actual.height, buckets, runtime.pipeline.vae.stem.stride * 8)
    settings = StreamingSettings(height=height, width=width, num_inference_steps=2, seed=request["seed"])
    # Official lossless transport gives uint8 RGB arrays rather than intermediate JPEGs.
    runtime.lossless_output = True
    with video_writer(output, actual.fps, width, height) as write:
        stream_frames(runtime, settings, Path(request["prompt_path"]).read_text().strip(),
                      decode_frames(source), actual.frame_count, actual.fps, write)
    validate_matching_timeline(actual, inspect_video_timeline(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
