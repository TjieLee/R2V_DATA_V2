from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any


def _load_pickle(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing official LR-ASD artifact: {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _detection_confidence(
    detections: list[list[dict[str, object]]],
    *,
    frame_index: int,
    bbox: list[float],
) -> float | None:
    if frame_index >= len(detections):
        return None
    matches = []
    for detection in detections[frame_index]:
        candidate = [float(value) for value in detection["bbox"]]  # type: ignore[index]
        matches.append((_iou(candidate, bbox), float(detection["conf"])))
    if not matches:
        return None
    overlap, confidence = max(matches, key=lambda item: (item[0], item[1]))
    return confidence if overlap >= 0.5 else None


def _video_metadata(path: Path, *, model_fps: float) -> tuple[int, int, float]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open official LR-ASD model video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError("official LR-ASD model video metadata is invalid")
    return width, height, frame_count / model_fps


def convert(args: argparse.Namespace) -> dict[str, object]:
    vendor_output = Path(args.vendor_output).expanduser().resolve(strict=True)
    work = vendor_output / "pywork"
    model_video = vendor_output / "pyavi" / "video.avi"
    audio = vendor_output / "pyavi" / "audio.wav"
    visualization = vendor_output / "pyavi" / "video_out.avi"
    if not audio.is_file():
        raise FileNotFoundError(f"missing official LR-ASD audio: {audio}")
    tracks = _load_pickle(work / "tracks.pckl")
    scores = _load_pickle(work / "scores.pckl")
    detections = _load_pickle(work / "faces.pckl")
    if len(tracks) != len(scores):
        raise ValueError("official LR-ASD tracks and score arrays do not align")
    model_fps = 25.0
    width, height, duration = _video_metadata(model_video, model_fps=model_fps)
    normalized_tracks: list[dict[str, object]] = []
    for track_index, (track_item, track_scores) in enumerate(
        zip(tracks, scores, strict=True),
        start=1,
    ):
        track = track_item["track"]
        frame_indices = [int(value) for value in track["frame"]]
        bboxes = [[float(value) for value in bbox] for bbox in track["bbox"]]
        logits = [float(value) for value in track_scores]
        if len(frame_indices) != len(bboxes):
            raise ValueError("official LR-ASD track frames and boxes differ")
        if len(logits) not in {len(frame_indices), len(frame_indices) - 1}:
            raise ValueError("official LR-ASD track frames and scores differ")
        scored_frame_indices = frame_indices[: len(logits)]
        scored_bboxes = bboxes[: len(logits)]
        samples = []
        for frame_index, bbox, logit in zip(
            scored_frame_indices,
            scored_bboxes,
            logits,
            strict=True,
        ):
            if not math.isfinite(logit):
                raise ValueError("official LR-ASD returned a non-finite score")
            samples.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index / model_fps,
                    "bbox_xyxy": bbox,
                    "detection_confidence": _detection_confidence(
                        detections,
                        frame_index=frame_index,
                        bbox=bbox,
                    ),
                    "raw_class1_logit": logit,
                    "backend_native_active": logit >= 0,
                }
            )
        normalized_tracks.append(
            {"face_track_id": f"face_{track_index}", "samples": samples}
        )
    return {
        "schema_version": "r2v.h3.lr_asd_native.1",
        "clip_uid": args.clip_uid,
        "source_video_path": str(Path(args.source_video).resolve(strict=True)),
        "model_video_path": str(model_video.resolve(strict=True)),
        "audio_path": str(audio.resolve(strict=True)),
        "official_visualization_path": (
            str(visualization.resolve(strict=True)) if visualization.is_file() else None
        ),
        "model_fps": model_fps,
        "audio_sample_rate_hz": 16000,
        "audio_channels": 1,
        "face_detector": "S3FD",
        "face_tracking": "shot_aware_iou",
        "face_crop_preprocessing": "official_lr_asd",
        "score_semantics": "lr_asd_native_class_1_logit",
        "active_decision_rule": "score_greater_than_or_equal_to_zero",
        "model_provenance": {
            "backend": "lr_asd",
            "model_identifier": "Junhua-Liao/LR-ASD",
            "checkpoint_path": str(Path(args.model_path).resolve(strict=True)),
            "checkpoint_sha256": args.checkpoint_sha256,
        },
        "width": width,
        "height": height,
        "duration_seconds": duration,
        "tracks": normalized_tracks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-uid", required=True)
    parser.add_argument("--source-video", required=True)
    parser.add_argument("--vendor-output", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    code_root = Path(args.code_root).resolve(strict=True)
    if not (code_root / "Columbia_test.py").is_file():
        raise ValueError("code root does not contain the official LR-ASD demo")
    payload = convert(args)
    output = Path(args.output).expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
