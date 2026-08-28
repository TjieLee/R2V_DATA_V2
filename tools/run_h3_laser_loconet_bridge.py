from __future__ import annotations

import argparse
import copy
import importlib.metadata
import io
import json
import math
import os
import subprocess
import sys
import tarfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

LASER_UPSTREAM_COMMIT = "3703d3f396cc7b29aa704364f8a9a5ab0c8c1fb9"
REQUIRED_STAGED_VENDOR_FILES = (
    "README.md",
    "LoCoNet/demoLoCoNet_landmark.py",
    "LoCoNet/landmark_loconet.py",
    "create_landmark.py",
)
LIP_LANDMARK_INDICES = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146, 61,
    76, 184, 74, 73, 72, 11, 302, 303, 304, 408, 306,
    307, 320, 404, 315, 16, 85, 180, 90, 77,
    62, 183, 42, 41, 38, 12, 268, 271, 272, 407, 292,
    325, 319, 403, 316, 15, 86, 179, 89, 96,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95, 78,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pinned LoCoNet+LASER preprocessing and inference",
    )
    parser.add_argument("--clip-uid", required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--landmark-model-path", type=Path, required=True)
    parser.add_argument("--s3fd-model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--landmark-model-sha256", required=True)
    parser.add_argument("--s3fd-model-sha256", required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _namespace(value: object) -> object:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _run(command: list[str], *, label: str) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with code {result.returncode}: {result.stderr.strip()}"
        )


def _equal_length_zip(
    *sequences: Sequence[Any],
    label: str,
) -> Iterator[tuple[Any, ...]]:
    lengths = [len(sequence) for sequence in sequences]
    if len(set(lengths)) > 1:
        raise ValueError(f"LASER {label} lengths differ: {lengths}")
    return zip(*sequences)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str, *, label: str) -> None:
    if _sha256(path) != expected:
        raise ValueError(f"LASER {label} SHA-256 does not match runtime provenance")


def _stage_pinned_vendor_source(
    *,
    code_root: Path,
    runtime_root: Path,
    upstream_commit: str,
) -> Path:
    destination = runtime_root / "vendor_source"
    if destination.exists():
        raise FileExistsError(f"LASER staged vendor source already exists: {destination}")
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(code_root),
                "archive",
                "--format=tar",
                upstream_commit,
                "LoCoNet",
                "create_landmark.py",
                "README.md",
            ],
            check=False,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"LASER pinned source archive failed: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"LASER pinned source archive failed with code {result.returncode}: {stderr}"
        )
    if not result.stdout:
        raise RuntimeError("LASER pinned source archive is empty")

    destination.mkdir(parents=False, exist_ok=False)
    try:
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as archive:
            members = archive.getmembers()
            if not members:
                raise RuntimeError("LASER pinned source archive is empty")
            destination_root = destination.resolve(strict=True)
            for member in members:
                member_path = (destination / member.name).resolve(strict=False)
                if not member_path.is_relative_to(destination_root):
                    raise RuntimeError("LASER pinned source archive contains unsafe path")
                if member.isdir():
                    member_path.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise RuntimeError(
                        "LASER pinned source archive contains unsupported entry"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(
                        "LASER pinned source archive contains unreadable file"
                    )
                member_path.parent.mkdir(parents=True, exist_ok=True)
                member_path.write_bytes(source.read())
    except (tarfile.TarError, OSError) as exc:
        raise RuntimeError(f"LASER pinned source extraction failed: {exc}") from exc

    missing = [
        relative
        for relative in REQUIRED_STAGED_VENDOR_FILES
        if not (destination / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            "LASER pinned source archive is missing required files: "
            + ", ".join(missing)
        )
    return destination


def _stage_s3fd_model(*, runtime_root: Path, source: Path) -> Path:
    destination = runtime_root / "model/faceDetector/s3fd/sfd_face.pth"
    destination.parent.mkdir(parents=True, exist_ok=False)
    destination.symlink_to(source)
    if destination.resolve(strict=True) != source.resolve(strict=True):
        raise ValueError("LASER S3FD runtime link does not resolve to explicit asset")
    return destination


def _install_gdown_blocker() -> None:
    blocker = ModuleType("gdown")

    def blocked_download(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("LASER runtime forbids gdown network downloads")

    blocker.download = blocked_download  # type: ignore[attr-defined]
    sys.modules["gdown"] = blocker


@contextmanager
def _legacy_numpy_s3fd_compatibility(numpy_module: object):
    # Pinned S3FD box_utils uses only the removed np.int scalar alias.
    namespace = numpy_module.__dict__  # type: ignore[attr-defined]
    had_int_alias = "int" in namespace
    original_int_alias = namespace.get("int")
    if not had_int_alias:
        numpy_module.int = int  # type: ignore[attr-defined]
    try:
        yield
    finally:
        if had_int_alias:
            numpy_module.int = original_int_alias  # type: ignore[attr-defined]
        else:
            del numpy_module.int  # type: ignore[attr-defined]


@contextmanager
def _offline_vggish_construction(torch_module: object, vggish_class: type[Any]):
    original_init = vggish_class.__init__
    original_download = torch_module.hub.load_state_dict_from_url  # type: ignore[attr-defined]

    def local_init(instance: object, *args: object, **kwargs: object) -> None:
        kwargs["pretrained"] = False
        original_init(instance, *args, **kwargs)

    def blocked_download(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("LASER runtime forbids torch.hub network downloads")

    vggish_class.__init__ = local_init  # type: ignore[method-assign]
    torch_module.hub.load_state_dict_from_url = blocked_download  # type: ignore[attr-defined]
    try:
        yield
    finally:
        vggish_class.__init__ = original_init  # type: ignore[method-assign]
        torch_module.hub.load_state_dict_from_url = original_download  # type: ignore[attr-defined]


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


def _bounded_bbox(
    bbox: list[float],
    *,
    width: int,
    height: int,
) -> list[float]:
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError("LASER returned an invalid track bbox")
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        raise ValueError("LASER returned a degenerate track bbox")
    clipped = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError("LASER track bbox is outside the model video")
    return clipped


def _video_metadata(path: Path) -> tuple[int, int, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open LASER model video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError("LASER model video metadata is invalid")
    return width, height, frame_count


def _stable_tracks(tracks: list[dict[str, object]]) -> list[dict[str, object]]:
    def key(track: dict[str, object]) -> tuple[object, ...]:
        frames = track["frame"]  # type: ignore[assignment]
        boxes = track["bbox"]  # type: ignore[assignment]
        return (
            int(frames[0]),
            int(frames[-1]),
            *(float(value) for value in boxes[0]),
            len(frames),
        )

    return sorted(tracks, key=key)


def _landmarks_for_crop(detector: object, crop: object) -> tuple[list[list[float]], bool]:
    import mediapipe as mp
    import numpy as np

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(crop))
    result = detector.detect(image)  # type: ignore[attr-defined]
    if len(result.face_landmarks) != 1:
        return [[-1.0, -1.0] for _ in LIP_LANDMARK_INDICES], False
    source = result.face_landmarks[0]
    values: list[list[float]] = []
    available = True
    for index in LIP_LANDMARK_INDICES:
        if index >= len(source):
            values.append([-1.0, -1.0])
            available = False
            continue
        landmark = source[index]
        x, y = float(landmark.x), float(landmark.y)
        if not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1:
            values.append([-1.0, -1.0])
            available = False
        else:
            values.append([x, y])
    return values, available


def _pad_sequence(
    value: object,
    *,
    input_start: int,
    input_end: int,
    output_start: int,
    output_end: int,
    fill_value: float,
) -> object:
    import torch

    tensor = value
    result = tensor[
        max(input_start, output_start) - input_start :
        min(input_end, output_end) - input_start + 1
    ]
    if input_start > output_start:
        shape = (input_start - output_start, *result.shape[1:])
        result = torch.cat(
            (torch.full(shape, fill_value, dtype=result.dtype), result), dim=0
        )
    if input_end < output_end:
        shape = (output_end - input_end, *result.shape[1:])
        result = torch.cat(
            (result, torch.full(shape, fill_value, dtype=result.dtype)), dim=0
        )
    return result


def _load_verified_checkpoint(
    model: object,
    path: Path,
    *,
    n_channel: int,
    layer: int,
) -> None:
    import torch

    loaded = torch.load(path, map_location="cpu")
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError("LASER checkpoint must contain a non-empty state dictionary")
    normalized: dict[str, object] = {}
    for raw_key, value in loaded.items():
        key = str(raw_key).replace("model.module.", "model.")
        if key in normalized:
            raise ValueError("LASER checkpoint key normalization is ambiguous")
        normalized[key] = value
    current = model.state_dict()  # type: ignore[attr-defined]
    parameters = dict(model.named_parameters())  # type: ignore[attr-defined]
    if not current or not parameters:
        raise ValueError("LASER model exposes no inference state")
    overlap = sorted(set(current).intersection(normalized))
    if not overlap:
        raise ValueError("LASER checkpoint has no keys matching the requested model")
    for component in ("landmark_bottleneck", "bottle_neck"):
        component_parameters = [
            key
            for key in parameters
            if key.startswith(f"{component}.") or f".{component}." in key
        ]
        if not component_parameters:
            raise ValueError(f"LASER model is missing required {component} parameters")
        missing_component = [
            key for key in component_parameters if key not in normalized
        ]
        if missing_component:
            raise ValueError(
                f"LASER checkpoint is missing required {component} parameters"
            )
    missing_parameters = sorted(set(parameters).difference(normalized))
    if missing_parameters:
        raise ValueError(
            "LASER checkpoint leaves trainable parameters uninitialized for "
            f"n_channel={n_channel}, layer={layer}"
        )
    missing_state = sorted(
        key
        for key in set(current).difference(normalized)
        if not key.endswith(".num_batches_tracked")
    )
    if missing_state:
        raise ValueError(
            "LASER checkpoint is missing inference state for "
            f"n_channel={n_channel}, layer={layer}"
        )
    shape_mismatch = [
        key for key in overlap if tuple(current[key].shape) != tuple(normalized[key].shape)
    ]
    if shape_mismatch:
        raise ValueError(
            "LASER checkpoint contains incompatible tensor shapes for "
            f"n_channel={n_channel}, layer={layer}"
        )
    unexpected_state = sorted(set(normalized).difference(current))
    if unexpected_state:
        raise ValueError("LASER checkpoint contains unexpected model state")
    result = model.load_state_dict(normalized, strict=False)  # type: ignore[attr-defined]
    if not hasattr(result, "missing_keys") or not hasattr(result, "unexpected_keys"):
        raise ValueError("LASER checkpoint load did not return torch incompatibility data")
    missing_after_load = [
        key
        for key in result.missing_keys
        if not key.endswith(".num_batches_tracked")
    ]
    if missing_after_load or result.unexpected_keys:
        raise ValueError("LASER checkpoint load returned incompatible model state")
    print(
        json.dumps(
            {
                "checkpoint_matching_key_count": len(overlap),
                "checkpoint_missing_key_count": len(missing_after_load),
                "checkpoint_unexpected_key_count": 0,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )


def _prepare_inputs(
    *,
    args: SimpleNamespace,
    tracks: list[dict[str, object]],
    detector: object,
    crop_thumbnail: object,
) -> tuple[list[object], list[object], list[object], list[list[bool]]]:
    import cv2
    import numpy as np
    import torch
    from scipy.io import wavfile
    from torchvggish import vggish_input

    visual_info: list[dict[str, Any]] = []
    availability: list[list[bool]] = []
    for track in tracks:
        crops = []
        landmarks = []
        track_availability = []
        for index, frame_number in enumerate(track["frame"]):  # type: ignore[index]
            frame = cv2.imread(
                str(Path(args.pyframesPath) / f"{int(frame_number) + 1:06}.jpg")
            )
            if frame is None:
                raise ValueError("LASER model frame is missing")
            crop, _ = crop_thumbnail(  # type: ignore[operator]
                frame,
                track["bbox"][index],  # type: ignore[index]
                padding=0.775,
                size=112,
            )
            values, available = _landmarks_for_crop(detector, crop)
            crops.append(torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)))
            landmarks.append(torch.tensor(values, dtype=torch.float32))
            track_availability.append(available)
        visual_info.append(
            {
                "frames": [int(value) for value in track["frame"]],  # type: ignore[index]
                "crops": torch.stack(crops),
                "landmarks": torch.stack(landmarks),
            }
        )
        availability.append(track_availability)

    visual_features: list[object] = []
    landmark_features: list[object] = []
    audio_features: list[object] = []
    for person_id, target in enumerate(visual_info):
        candidates = []
        target_frames = target["frames"]
        for candidate_id, candidate in enumerate(visual_info):
            if candidate_id == person_id:
                continue
            overlap = set(candidate["frames"]) & set(target_frames)
            if len(overlap) >= len(target_frames) / 2:
                candidates.append(
                    (
                        candidate["frames"][0],
                        candidate["frames"][-1],
                        candidate_id,
                    )
                )
        candidates.sort()
        chosen = [] if not candidates else [candidates[0]]
        if len(candidates) > 1:
            chosen.append(candidates[-1])
        contexts = []
        context_landmarks = []
        for start, end, candidate_id in chosen:
            candidate = visual_info[candidate_id]
            contexts.append(
                _pad_sequence(
                    candidate["crops"],
                    input_start=start,
                    input_end=end,
                    output_start=target_frames[0],
                    output_end=target_frames[-1],
                    fill_value=0.0,
                )
            )
            context_landmarks.append(
                _pad_sequence(
                    candidate["landmarks"],
                    input_start=start,
                    input_end=end,
                    output_start=target_frames[0],
                    output_end=target_frames[-1],
                    fill_value=-1.0,
                )
            )
        while len(contexts) < 2:
            contexts.append(target["crops"])
            context_landmarks.append(target["landmarks"])
        visual_features.append(
            torch.stack((target["crops"], contexts[0], contexts[1])).unsqueeze(0)
        )
        landmark_features.append(
            torch.stack(
                (target["landmarks"], context_landmarks[0], context_landmarks[1])
            ).unsqueeze(0)
        )
        track_audio = Path(args.pycropPath) / f"audio{person_id:06d}.wav"
        start = target_frames[0] / 25
        end = target_frames[-1] / 25
        if end <= start:
            raise ValueError("LASER face track is too short for audio inference")
        _run(
            [
                args.ffmpeg,
                "-y",
                "-i",
                args.audioFilePath,
                "-ac",
                "1",
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ss",
                f"{start:.3f}",
                "-to",
                f"{end:.3f}",
                str(track_audio),
                "-loglevel",
                "error",
            ],
            label="LASER track audio extraction",
        )
        sample_rate, waveform = wavfile.read(track_audio)
        audio = vggish_input.waveform_to_examples(
            np.asarray(waveform), sample_rate, len(target_frames), 25, False
        )
        audio_features.append(torch.from_numpy(audio).unsqueeze(0).unsqueeze(0))
    return visual_features, landmark_features, audio_features, availability


def _render_visualization(
    *,
    model_video: Path,
    audio_path: Path,
    tracks: list[dict[str, object]],
    scores: list[list[float]],
    output: Path,
    ffmpeg: str,
) -> None:
    import cv2

    capture = cv2.VideoCapture(str(model_video))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    silent = output.with_name("laser_visualization_silent.mp4")
    writer = cv2.VideoWriter(
        str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    by_frame: dict[int, list[tuple[list[float], float, int]]] = {}
    for track_index, (track, track_scores) in enumerate(
        _equal_length_zip(tracks, scores, label="visualization track/score")
    ):
        for frame_index, bbox, score in _equal_length_zip(
            track["frame"],  # type: ignore[arg-type]
            track["bbox"],  # type: ignore[arg-type]
            track_scores,
            label="visualization frame/bbox/score",
        ):
            by_frame.setdefault(int(frame_index), []).append(
                ([float(value) for value in bbox], float(score), track_index + 1)
            )
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        for bbox, score, track_id in by_frame.get(frame_index, []):
            x1, y1, x2, y2 = (round(value) for value in bbox)
            color = (0, 220, 0) if score >= 0 else (0, 0, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"face_{track_id} {score:+.3f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()
    _run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(silent),
            "-i",
            str(audio_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-shortest",
            "-loglevel",
            "error",
            str(output),
        ],
        label="LASER visualization mux",
    )


def run(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.upstream_commit != LASER_UPSTREAM_COMMIT:
        raise ValueError("LASER bridge only supports the pinned upstream commit")
    if not arguments.device.startswith("cuda:"):
        raise ValueError("LoCoNet+LASER bridge requires an explicit cuda:N device")
    source = arguments.source_video.expanduser().resolve(strict=True)
    code_root = arguments.code_root.expanduser().resolve(strict=True)
    model_path = arguments.model_path.expanduser().resolve(strict=True)
    config_path = arguments.config_path.expanduser().resolve(strict=True)
    landmark_model_path = arguments.landmark_model_path.expanduser().resolve(
        strict=True
    )
    s3fd_model_path = arguments.s3fd_model_path.expanduser().resolve(strict=True)
    work = arguments.work_dir.expanduser().resolve(strict=True)
    for path, expected, label in (
        (model_path, arguments.checkpoint_sha256, "model checkpoint"),
        (config_path, arguments.config_sha256, "config"),
        (landmark_model_path, arguments.landmark_model_sha256, "landmark model"),
        (s3fd_model_path, arguments.s3fd_model_sha256, "S3FD model"),
    ):
        _require_sha256(path, expected, label=label)
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices:
        raise ValueError("LASER subprocess requires isolated CUDA_VISIBLE_DEVICES")
    runtime_root = work / "vendor_runtime"
    runtime_root.mkdir(exist_ok=False)
    vendor_source = _stage_pinned_vendor_source(
        code_root=code_root,
        runtime_root=runtime_root,
        upstream_commit=arguments.upstream_commit,
    )
    loconet_root = vendor_source / "LoCoNet"
    _stage_s3fd_model(runtime_root=runtime_root, source=s3fd_model_path)
    _install_gdown_blocker()
    sys.path.insert(0, str(vendor_source))
    sys.path.insert(0, str(loconet_root))
    os.chdir(runtime_root)

    import numpy as np
    import torch
    import yaml

    with _legacy_numpy_s3fd_compatibility(np):
        import demoLoCoNet_landmark as vendor_demo
        from landmark_loconet import loconet
        from torchvggish.vggish import VGGish

    try:
        import mediapipe
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("LASER runtime requires mediapipe") from exc

    try:
        mediapipe_version = importlib.metadata.version("mediapipe")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("LASER runtime requires an installed mediapipe package") from exc
    if not mediapipe_version.strip() or not hasattr(mediapipe, "Image"):
        raise RuntimeError("LASER runtime could not initialize mediapipe")
    torch_version = str(torch.__version__)

    device_index = int(arguments.device.split(":", 1)[1])
    torch.cuda.set_device(device_index)
    pyavi = work / "pyavi"
    frames = work / "pyframes"
    pywork = work / "pywork"
    pycrop = work / "pycrop"
    for directory in (pyavi, frames, pywork, pycrop):
        directory.mkdir(exist_ok=True)
    model_video = pyavi / "video.avi"
    audio_path = pyavi / "audio.wav"
    _run(
        [
            arguments.ffmpeg,
            "-y",
            "-i",
            str(source),
            "-qscale:v",
            "2",
            "-threads",
            "1",
            "-r",
            "25",
            str(model_video),
            "-loglevel",
            "error",
        ],
        label="LASER model-video extraction",
    )
    _run(
        [
            arguments.ffmpeg,
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-vn",
            "-ar",
            "16000",
            str(audio_path),
            "-loglevel",
            "error",
        ],
        label="LASER audio extraction",
    )
    _run(
        [
            arguments.ffmpeg,
            "-y",
            "-i",
            str(model_video),
            "-qscale:v",
            "2",
            "-threads",
            "1",
            "-f",
            "image2",
            str(frames / "%06d.jpg"),
            "-loglevel",
            "error",
        ],
        label="LASER frame extraction",
    )
    vendor_args = SimpleNamespace(
        videoFilePath=str(model_video),
        audioFilePath=str(audio_path),
        pyframesPath=str(frames),
        pyworkPath=str(pywork),
        pycropPath=str(pycrop),
        nDataLoaderThread=1,
        numFailedDet=12,
        minTrack=20,
        minFaceSize=1,
        ffmpeg=arguments.ffmpeg,
    )
    with _legacy_numpy_s3fd_compatibility(np):
        scenes = vendor_demo.scene_detect(vendor_args)
        detections = vendor_demo.inference_video(vendor_args)
        detection_evidence = copy.deepcopy(detections)
        tracks = []
        for shot in scenes:
            if shot[1].frame_num - shot[0].frame_num >= vendor_args.minTrack:
                tracks.extend(
                    vendor_demo.track_shot(
                        vendor_args,
                        detections[shot[0].frame_num : shot[1].frame_num],
                    )
                )
    tracks = _stable_tracks(tracks)

    base_options = mp_python.BaseOptions(
        model_asset_path=str(landmark_model_path),
        delegate=mp_python.BaseOptions.Delegate.GPU,
    )
    detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
    )
    try:
        visual, landmarks, audio, availability = _prepare_inputs(
            args=vendor_args,
            tracks=tracks,
            detector=detector,
            crop_thumbnail=vendor_demo.crop_thumbnail,
        )
    finally:
        detector.close()

    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise TypeError("LASER config must be a YAML object")
    try:
        resolved_n_channel = int(config_payload["n_channel"])
        resolved_layer = int(config_payload["layer"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("LASER config requires integer n_channel and layer") from exc
    if resolved_n_channel <= 0 or resolved_layer < 0:
        raise ValueError(
            "LASER config n_channel must be positive and layer non-negative"
        )
    config_payload["WORKSPACE"] = str(work / "model_workspace")
    cfg = _namespace(config_payload)
    with _offline_vggish_construction(torch, VGGish):
        model = loconet(
            cfg,
            n_channel=resolved_n_channel,
            layer=resolved_layer,
        )
    _load_verified_checkpoint(
        model,
        model_path,
        n_channel=resolved_n_channel,
        layer=resolved_layer,
    )
    model = model.to(device=arguments.device)
    model.eval()
    scores: list[list[float]] = []
    with torch.no_grad():
        for visual_item, landmark_item, audio_item in _equal_length_zip(
            visual,
            landmarks,
            audio,
            label="visual/landmark/audio input",
        ):
            prediction = model.model.forward_evaluation(
                audio_item.to(dtype=torch.float, device=arguments.device),
                visual_item.to(dtype=torch.float, device=arguments.device),
                landmark_item.to(dtype=torch.float, device=arguments.device),
                None,
                None,
                True,
            )
            values = [float(value) for value in prediction.detach().cpu().tolist()]
            scores.append(values)
    if len(scores) != len(tracks):
        raise ValueError("LASER score arrays do not align with face tracks")

    width, height, frame_count = _video_metadata(model_video)
    normalized_tracks = []
    landmark_sample_count = 0
    landmark_available_count = 0
    for track_index, (track, track_scores, track_landmarks) in enumerate(
        _equal_length_zip(
            tracks,
            scores,
            availability,
            label="track/score/landmark availability",
        ),
        start=1,
    ):
        frame_indices = [int(value) for value in track["frame"]]  # type: ignore[index]
        bboxes = [
            [float(value) for value in bbox]
            for bbox in track["bbox"]  # type: ignore[index]
        ]
        samples = []
        for frame_index, raw_bbox, score, landmark_available in _equal_length_zip(
            frame_indices,
            bboxes,
            track_scores,
            track_landmarks,
            label="track frame/bbox/score/landmark sample",
        ):
            if not math.isfinite(score):
                raise ValueError("LASER returned a non-finite native score")
            samples.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": frame_index / 25.0,
                    "bbox_xyxy": _bounded_bbox(
                        raw_bbox, width=width, height=height
                    ),
                    "detection_confidence": _detection_confidence(
                        detection_evidence,
                        frame_index=frame_index,
                        bbox=raw_bbox,
                    ),
                    "raw_backend_score": score,
                    "backend_native_active": score >= 0,
                    "landmark_available": landmark_available,
                }
            )
        landmark_sample_count += len(samples)
        landmark_available_count += sum(track_landmarks)
        normalized_tracks.append(
            {"face_track_id": f"face_{track_index}", "samples": samples}
        )

    visualization = work / "laser_visualization.mp4"
    _render_visualization(
        model_video=model_video,
        audio_path=audio_path,
        tracks=tracks,
        scores=scores,
        output=visualization,
        ffmpeg=arguments.ffmpeg,
    )
    return {
        "schema_version": "r2v.h3.laser_asd_native.2",
        "clip_uid": arguments.clip_uid,
        "source_video_path": str(source),
        "model_video_path": str(model_video.resolve(strict=True)),
        "audio_path": str(audio_path.resolve(strict=True)),
        "debug_visualization_path": str(visualization.resolve(strict=True)),
        "model_fps": 25.0,
        "audio_sample_rate_hz": 16000,
        "audio_channels": 1,
        "face_detector": "S3FD",
        "face_tracking": "shot_aware_iou",
        "face_crop_preprocessing": "laser_loconet_official_face_crop",
        "landmark_backend": "mediapipe_face_landmarker_82_lips",
        "score_semantics": "laser_loconet_native_score",
        "active_decision_rule": "score_greater_than_or_equal_to_zero",
        "upstream_repository": "https://github.com/plnguyen2908/LASER_ASD",
        "upstream_commit": LASER_UPSTREAM_COMMIT,
        "model_provenance": {
            "backend": "laser_asd_loconet",
            "model_identifier": "plnguyen2908/LASER_ASD LoCoNet+LASER",
            "checkpoint_path": str(model_path),
            "checkpoint_sha256": arguments.checkpoint_sha256,
        },
        "config_path": str(config_path),
        "config_sha256": arguments.config_sha256,
        "landmark_model_path": str(landmark_model_path),
        "landmark_model_sha256": arguments.landmark_model_sha256,
        "s3fd_model_path": str(s3fd_model_path),
        "s3fd_model_sha256": arguments.s3fd_model_sha256,
        "resolved_n_channel": resolved_n_channel,
        "resolved_layer": resolved_layer,
        "device": arguments.device,
        "cuda_visible_devices": visible_devices,
        "mediapipe_version": mediapipe_version,
        "torch_version": torch_version,
        "width": width,
        "height": height,
        "duration_seconds": frame_count / 25.0,
        "landmark_sample_count": landmark_sample_count,
        "landmark_available_count": landmark_available_count,
        "deterministic_context_selection": "stable_first_last",
        "tracks": normalized_tracks,
    }


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = run(arguments)
    output = arguments.output.expanduser().resolve(strict=False)
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
