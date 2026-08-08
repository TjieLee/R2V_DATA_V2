from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_SCRFD_FILENAME = "scrfd_10g_bnkps.onnx"
_LANDMARKER_FILENAME = "face_landmarker_v2_with_blendshapes.task"
_DETECTION_THRESHOLD = 0.5
_NMS_THRESHOLD = 0.4
_SCRFD_STRIDES = (8, 16, 32)
_DYNAMIC_SCRFD_INPUT_SIZE = 640


def _load_runtime_dependencies() -> tuple[Any, Any]:
    import mediapipe
    import onnxruntime

    return onnxruntime, mediapipe


@dataclass(frozen=True)
class _FaceDetection:
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    landmarks_5: tuple[tuple[float, float], ...]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _select_face(detections: list[_FaceDetection]) -> _FaceDetection | None:
    if not detections:
        return None
    return min(
        detections,
        key=lambda detection: (
            -detection.area,
            -detection.confidence,
            detection.bbox_xyxy,
        ),
    )


def _intersection_over_union(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(
        0.0,
        min(left_y2, right_y2) - max(left_y1, right_y1),
    )
    intersection = intersection_width * intersection_height
    union = (
        max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
        + max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _non_maximum_suppression(
    detections: list[_FaceDetection],
) -> list[_FaceDetection]:
    ordered = sorted(
        detections,
        key=lambda detection: (
            -detection.confidence,
            -detection.area,
            detection.bbox_xyxy,
        ),
    )
    retained: list[_FaceDetection] = []
    for detection in ordered:
        if all(
            _intersection_over_union(detection.bbox_xyxy, item.bbox_xyxy)
            <= _NMS_THRESHOLD
            for item in retained
        ):
            retained.append(detection)
    return retained


def _distance_to_bbox(
    centers: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    return np.stack(
        (
            centers[:, 0] - distances[:, 0],
            centers[:, 1] - distances[:, 1],
            centers[:, 0] + distances[:, 2],
            centers[:, 1] + distances[:, 3],
        ),
        axis=1,
    )


def _distance_to_landmarks(
    centers: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    landmarks = distances.reshape(-1, 5, 2).copy()
    landmarks[..., 0] += centers[:, None, 0]
    landmarks[..., 1] += centers[:, None, 1]
    return landmarks


def _json_safe_onnx_dimension(value: object) -> int | str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return str(value)


def _is_static_integer_dimension(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


class _ScrfdDetector:
    def __init__(self, model_path: Path, onnxruntime: Any) -> None:
        self.session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError("SCRFD must expose exactly one image input")
        shape = tuple(inputs[0].shape)
        if len(shape) != 4:
            raise ValueError("SCRFD input must have rank-four NCHW dimensions")
        batch, channels, declared_height, declared_width = shape
        if _is_static_integer_dimension(batch) and batch != 1:
            raise ValueError("SCRFD static batch dimension must equal one")
        if _is_static_integer_dimension(channels) and channels != 3:
            raise ValueError("SCRFD static channel dimension must equal three")
        for dimension in (declared_height, declared_width):
            if _is_static_integer_dimension(dimension) and dimension <= 0:
                raise ValueError("SCRFD static spatial dimensions must be positive")

        self.model_input_shape = [
            _json_safe_onnx_dimension(value) for value in shape
        ]
        self.dynamic_spatial = not (
            _is_static_integer_dimension(declared_height)
            and _is_static_integer_dimension(declared_width)
        )
        if self.dynamic_spatial:
            self.input_height = _DYNAMIC_SCRFD_INPUT_SIZE
            self.input_width = _DYNAMIC_SCRFD_INPUT_SIZE
        else:
            self.input_height = int(declared_height)
            self.input_width = int(declared_width)
        self.input_name = str(inputs[0].name)
        self.provider = "CPUExecutionProvider"

    def _preprocess(self, image: Image.Image) -> tuple[np.ndarray, float]:
        rgb = image.convert("RGB")
        scale = min(self.input_width / rgb.width, self.input_height / rgb.height)
        resized_size = (
            max(1, round(rgb.width * scale)),
            max(1, round(rgb.height * scale)),
        )
        resized = rgb.resize(resized_size, Image.Resampling.BILINEAR)
        canvas = np.zeros((self.input_height, self.input_width, 3), dtype=np.float32)
        canvas[: resized_size[1], : resized_size[0]] = np.asarray(
            resized,
            dtype=np.float32,
        )
        tensor = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None]
        return np.ascontiguousarray(tensor, dtype=np.float32), scale

    def detect(self, image: Image.Image) -> list[_FaceDetection]:
        tensor, scale = self._preprocess(image)
        output_values = self.session.run(None, {self.input_name: tensor})
        if len(output_values) not in {6, 9}:
            raise ValueError("SCRFD output count must be six or nine")
        feature_levels = len(output_values) // (3 if len(output_values) == 9 else 2)
        if feature_levels != len(_SCRFD_STRIDES):
            raise ValueError("SCRFD output feature levels are unsupported")
        score_outputs = output_values[:feature_levels]
        bbox_outputs = output_values[feature_levels : feature_levels * 2]
        landmark_outputs = (
            output_values[feature_levels * 2 :] if len(output_values) == 9 else None
        )
        detections: list[_FaceDetection] = []
        for level, stride in enumerate(_SCRFD_STRIDES):
            scores = np.asarray(score_outputs[level], dtype=np.float32).reshape(-1)
            boxes = np.asarray(bbox_outputs[level], dtype=np.float32).reshape(-1, 4)
            feature_height = self.input_height // stride
            feature_width = self.input_width // stride
            location_count = feature_height * feature_width
            if location_count <= 0 or scores.size % location_count:
                raise ValueError("SCRFD score geometry is invalid")
            anchor_count = scores.size // location_count
            if boxes.shape[0] != scores.size or anchor_count <= 0:
                raise ValueError("SCRFD bbox geometry is invalid")
            grid_x, grid_y = np.meshgrid(
                np.arange(feature_width, dtype=np.float32),
                np.arange(feature_height, dtype=np.float32),
            )
            centers = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2) * stride
            centers = np.repeat(centers, anchor_count, axis=0)
            decoded_boxes = _distance_to_bbox(centers, boxes * stride) / scale
            if landmark_outputs is not None:
                distances = np.asarray(
                    landmark_outputs[level],
                    dtype=np.float32,
                ).reshape(-1, 10)
                if distances.shape[0] != scores.size:
                    raise ValueError("SCRFD landmark geometry is invalid")
                decoded_landmarks = (
                    _distance_to_landmarks(centers, distances * stride) / scale
                )
            else:
                decoded_landmarks = np.empty((scores.size, 0, 2), dtype=np.float32)
            for index in np.flatnonzero(scores >= _DETECTION_THRESHOLD):
                x1, y1, x2, y2 = decoded_boxes[index]
                clipped = (
                    float(np.clip(x1, 0, image.width)),
                    float(np.clip(y1, 0, image.height)),
                    float(np.clip(x2, 0, image.width)),
                    float(np.clip(y2, 0, image.height)),
                )
                if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                    continue
                landmarks = tuple(
                    (float(point[0]), float(point[1]))
                    for point in decoded_landmarks[index]
                )
                detections.append(
                    _FaceDetection(
                        confidence=float(scores[index]),
                        bbox_xyxy=clipped,
                        landmarks_5=landmarks,
                    )
                )
        return _non_maximum_suppression(detections)


def _create_face_landmarker(mediapipe: Any, model_path: Path) -> Any:
    base_options = mediapipe.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mediapipe.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mediapipe.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return mediapipe.tasks.vision.FaceLandmarker.create_from_options(options)


def _euler_degrees(matrix: np.ndarray) -> tuple[float, float, float]:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape not in {(3, 3), (4, 4)} or not np.isfinite(value).all():
        raise ValueError("facial transformation matrix must be finite 3x3 or 4x4")
    rotation = value[:3, :3]
    sin_yaw = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    yaw = math.asin(sin_yaw)
    cosine_yaw = math.cos(yaw)
    if abs(cosine_yaw) > 1e-6:
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
        roll = 0.0
    return tuple(math.degrees(angle) for angle in (yaw, pitch, roll))


def _face_crop(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    margin = 0.25 * max(x2 - x1, y2 - y1)
    crop_box = (
        max(0, math.floor(x1 - margin)),
        max(0, math.floor(y1 - margin)),
        min(image.width, math.ceil(x2 + margin)),
        min(image.height, math.ceil(y2 + margin)),
    )
    return image.crop(crop_box).convert("RGB")


class _ScrfdMediaPipeScorer:
    def __init__(
        self,
        *,
        detector: _ScrfdDetector,
        landmarker: Any,
        mediapipe: Any,
        model_root: Path,
    ) -> None:
        self.detector = detector
        self.landmarker = landmarker
        self.mediapipe = mediapipe
        self.model_root = model_root

    def eval(self) -> _ScrfdMediaPipeScorer:
        return self

    def inspect(self, image: Image.Image) -> dict[str, object]:
        started = time.monotonic()
        rgb = image.convert("RGB")
        raw_metrics: dict[str, object] = {
            "provider": self.detector.provider,
            "device": "cpu",
            "scrfd_model_input_shape": self.detector.model_input_shape,
            "scrfd_dynamic_spatial": self.detector.dynamic_spatial,
            "scrfd_inference_width": self.detector.input_width,
            "scrfd_inference_height": self.detector.input_height,
            "angle_unit": "degree",
            "angle_convention": (
                "R=Rz(roll)Ry(yaw)Rx(pitch); positive angles follow the "
                "right-hand rule in MediaPipe facial-transform coordinates"
            ),
        }
        scrfd_started = time.monotonic()
        try:
            detections = self.detector.detect(rgb)
        except Exception as exc:  # noqa: BLE001 - report SCRFD request failure
            scrfd_runtime = time.monotonic() - scrfd_started
            raw_metrics["scrfd_error"] = f"{type(exc).__name__}: {exc}"
            return {
                "face_detected": False,
                "face_count": 0,
                "face_detection_confidence": None,
                "face_bbox_area_ratio": None,
                "head_visible": None,
                "yaw": None,
                "pitch": None,
                "roll": None,
                "scrfd_success": False,
                "face_landmarker_success": None,
                "scrfd_runtime_seconds": scrfd_runtime,
                "face_landmarker_runtime_seconds": None,
                "runtime_seconds": time.monotonic() - started,
                "raw_metrics": raw_metrics,
            }
        scrfd_runtime = time.monotonic() - scrfd_started
        selected = _select_face(detections)
        raw_metrics["face_count"] = len(detections)
        if selected is None:
            return {
                "face_detected": False,
                "face_count": 0,
                "face_detection_confidence": None,
                "face_bbox_area_ratio": None,
                "head_visible": None,
                "yaw": None,
                "pitch": None,
                "roll": None,
                "scrfd_success": True,
                "face_landmarker_success": None,
                "scrfd_runtime_seconds": scrfd_runtime,
                "face_landmarker_runtime_seconds": None,
                "runtime_seconds": time.monotonic() - started,
                "raw_metrics": raw_metrics,
            }

        bbox = selected.bbox_xyxy
        raw_metrics["selected_face_bbox"] = list(bbox)
        raw_metrics["scrfd_landmarks_5"] = [
            list(point) for point in selected.landmarks_5
        ]
        area_ratio = selected.area / (rgb.width * rgb.height)
        center_x = (bbox[0] + bbox[2]) / (2 * rgb.width)
        center_y = (bbox[1] + bbox[3]) / (2 * rgb.height)
        yaw: float | None = None
        pitch: float | None = None
        roll: float | None = None
        transformation: list[list[float]] | None = None
        landmarker_success = False
        landmarker_started = time.monotonic()
        try:
            face_crop = np.asarray(_face_crop(rgb, bbox), dtype=np.uint8)
            media_image = self.mediapipe.Image(
                image_format=self.mediapipe.ImageFormat.SRGB,
                data=face_crop,
            )
            result = self.landmarker.detect(media_image)
            matrices = getattr(result, "facial_transformation_matrixes", None)
            if matrices:
                matrix = np.asarray(matrices[0], dtype=np.float64)
                yaw, pitch, roll = _euler_degrees(matrix)
                transformation = matrix.tolist()
                landmarker_success = True
        except Exception as exc:  # noqa: BLE001 - isolate landmarker from SCRFD
            raw_metrics["face_landmarker_error"] = f"{type(exc).__name__}: {exc}"
        landmarker_runtime = time.monotonic() - landmarker_started
        raw_metrics["facial_transformation_matrix"] = transformation
        return {
            "face_detected": True,
            "face_count": len(detections),
            "face_detection_confidence": selected.confidence,
            "face_bbox_area_ratio": area_ratio,
            "head_visible": True if landmarker_success else None,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "scrfd_success": True,
            "face_landmarker_success": landmarker_success,
            "scrfd_runtime_seconds": scrfd_runtime,
            "face_landmarker_runtime_seconds": landmarker_runtime,
            "runtime_seconds": time.monotonic() - started,
            "raw_metrics": {
                **raw_metrics,
                "face_center_x_norm": center_x,
                "face_center_y_norm": center_y,
            },
        }


def load_scorer(
    *,
    kind: str,
    backend: str,
    model_path: Path,
    local_files_only: bool,
) -> _ScrfdMediaPipeScorer:
    if kind != "subject_pose":
        raise ValueError("subject pose adapter only supports kind=subject_pose")
    if backend != "scrfd_mediapipe":
        raise ValueError(f"unsupported subject pose backend: {backend}")
    if local_files_only is not True:
        raise ValueError("subject pose adapter requires local_files_only")
    model_root = Path(model_path).expanduser().resolve(strict=True)
    if not model_root.is_dir():
        raise ValueError("subject pose model path must be a directory")
    scrfd_path = (model_root / _SCRFD_FILENAME).resolve(strict=True)
    landmarker_path = (model_root / _LANDMARKER_FILENAME).resolve(strict=True)
    if scrfd_path.parent != model_root or landmarker_path.parent != model_root:
        raise ValueError("subject pose models must be direct children of model root")
    onnxruntime, mediapipe = _load_runtime_dependencies()
    detector = _ScrfdDetector(scrfd_path, onnxruntime)
    landmarker = _create_face_landmarker(mediapipe, landmarker_path)
    return _ScrfdMediaPipeScorer(
        detector=detector,
        landmarker=landmarker,
        mediapipe=mediapipe,
        model_root=model_root,
    )
