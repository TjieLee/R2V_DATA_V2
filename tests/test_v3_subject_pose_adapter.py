from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tools.reference_filter_adapters.subject_pose import (
    r2v_reference_filter_adapter as adapter,
)


class FakeMediaPipe:
    class ImageFormat:
        SRGB = "srgb"

    class Image:
        def __init__(self, *, image_format: object, data: np.ndarray) -> None:
            self.image_format = image_format
            self.data = data


class FakeDetector:
    provider = "CPUExecutionProvider"

    def __init__(
        self,
        detections: list[adapter._FaceDetection] | Exception,
    ) -> None:
        self.detections = detections
        self.calls = 0

    def detect(self, image: Image.Image) -> list[adapter._FaceDetection]:
        assert image.mode == "RGB"
        self.calls += 1
        if isinstance(self.detections, Exception):
            raise self.detections
        return self.detections


class FakeLandmarker:
    def __init__(self, matrix: np.ndarray | Exception | None) -> None:
        self.matrix = matrix
        self.calls = 0

    def detect(self, image: object) -> object:
        assert isinstance(image, FakeMediaPipe.Image)
        self.calls += 1
        if isinstance(self.matrix, Exception):
            raise self.matrix
        matrices = [] if self.matrix is None else [self.matrix]
        return SimpleNamespace(facial_transformation_matrixes=matrices)


def _detection(
    confidence: float,
    bbox: tuple[float, float, float, float],
) -> adapter._FaceDetection:
    return adapter._FaceDetection(
        confidence=confidence,
        bbox_xyxy=bbox,
        landmarks_5=((11.0, 12.0),) * 5,
    )


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    yaw_radians, pitch_radians, roll_radians = map(
        math.radians,
        (yaw, pitch, roll),
    )
    rotate_x = np.array(
        (
            (1.0, 0.0, 0.0),
            (0.0, math.cos(pitch_radians), -math.sin(pitch_radians)),
            (0.0, math.sin(pitch_radians), math.cos(pitch_radians)),
        )
    )
    rotate_y = np.array(
        (
            (math.cos(yaw_radians), 0.0, math.sin(yaw_radians)),
            (0.0, 1.0, 0.0),
            (-math.sin(yaw_radians), 0.0, math.cos(yaw_radians)),
        )
    )
    rotate_z = np.array(
        (
            (math.cos(roll_radians), -math.sin(roll_radians), 0.0),
            (math.sin(roll_radians), math.cos(roll_radians), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotate_z @ rotate_y @ rotate_x
    return matrix


def _scorer(
    detections: list[adapter._FaceDetection] | Exception,
    matrix: np.ndarray | Exception | None,
) -> tuple[adapter._ScrfdMediaPipeScorer, FakeDetector, FakeLandmarker]:
    detector = FakeDetector(detections)
    landmarker = FakeLandmarker(matrix)
    scorer = adapter._ScrfdMediaPipeScorer(
        detector=detector,
        landmarker=landmarker,
        mediapipe=FakeMediaPipe,
        model_root=Path("models"),
    )
    return scorer, detector, landmarker


def test_face_selection_is_deterministic_and_pose_uses_degrees() -> None:
    small_high_confidence = _detection(0.99, (10.0, 10.0, 30.0, 30.0))
    large_face = _detection(0.85, (35.0, 20.0, 85.0, 80.0))
    matrix = _rotation_matrix(yaw=25.0, pitch=-7.0, roll=4.0)
    scorer, detector, landmarker = _scorer(
        [small_high_confidence, large_face],
        matrix,
    )

    result = scorer.inspect(Image.new("RGB", (100, 100), "white"))

    assert detector.calls == 1
    assert landmarker.calls == 1
    assert result["face_detected"] is True
    assert result["face_count"] == 2
    assert result["face_detection_confidence"] == 0.85
    assert result["face_bbox_area_ratio"] == pytest.approx(0.3)
    assert result["yaw"] == pytest.approx(25.0)
    assert result["pitch"] == pytest.approx(-7.0)
    assert result["roll"] == pytest.approx(4.0)
    assert result["head_visible"] is True
    assert result["raw_metrics"]["selected_face_bbox"] == [35.0, 20.0, 85.0, 80.0]
    assert result["raw_metrics"]["provider"] == "CPUExecutionProvider"
    assert result["raw_metrics"]["device"] == "cpu"


def test_no_face_does_not_call_landmarker_or_invent_head_visibility() -> None:
    scorer, _, landmarker = _scorer([], np.eye(4))

    result = scorer.inspect(Image.new("RGB", (80, 60), "white"))

    assert result["face_detected"] is False
    assert result["face_count"] == 0
    assert result["head_visible"] is None
    assert result["scrfd_success"] is True
    assert result["face_landmarker_success"] is None
    assert landmarker.calls == 0


def test_landmarker_failure_is_isolated_after_scrfd_detection() -> None:
    scorer, _, _ = _scorer(
        [_detection(0.9, (10.0, 8.0, 50.0, 52.0))],
        RuntimeError("landmarker unavailable"),
    )

    result = scorer.inspect(Image.new("RGB", (80, 60), "white"))

    assert result["face_detected"] is True
    assert result["scrfd_success"] is True
    assert result["face_landmarker_success"] is False
    assert result["head_visible"] is None
    assert result["yaw"] is None
    assert "landmarker unavailable" in result["raw_metrics"][
        "face_landmarker_error"
    ]


def test_scrfd_failure_is_reported_without_fake_no_face_success() -> None:
    scorer, _, landmarker = _scorer(RuntimeError("onnx failure"), np.eye(4))

    result = scorer.inspect(Image.new("RGB", (80, 60), "white"))

    assert result["face_detected"] is False
    assert result["scrfd_success"] is False
    assert result["face_landmarker_success"] is None
    assert result["head_visible"] is None
    assert landmarker.calls == 0
    assert "onnx failure" in result["raw_metrics"]["scrfd_error"]


def test_load_scorer_uses_exact_local_paths_and_cpu_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root = tmp_path / "face-models"
    model_root.mkdir()
    scrfd_path = model_root / "scrfd_10g_bnkps.onnx"
    landmarker_path = model_root / "face_landmarker_v2_with_blendshapes.task"
    scrfd_path.write_bytes(b"onnx")
    landmarker_path.write_bytes(b"task")
    session_calls: list[tuple[str, list[str]]] = []
    landmarker_calls: list[Path] = []

    class FakeSession:
        def get_inputs(self) -> list[object]:
            return [SimpleNamespace(shape=[1, 3, 640, 640], name="input.1")]

    class FakeOnnxRuntime:
        @staticmethod
        def InferenceSession(path: str, *, providers: list[str]) -> FakeSession:
            session_calls.append((path, providers))
            return FakeSession()

    def fake_create_landmarker(_mediapipe: object, path: Path) -> FakeLandmarker:
        landmarker_calls.append(path)
        return FakeLandmarker(np.eye(4))

    monkeypatch.setattr(
        adapter,
        "_load_runtime_dependencies",
        lambda: (FakeOnnxRuntime, FakeMediaPipe),
    )
    monkeypatch.setattr(adapter, "_create_face_landmarker", fake_create_landmarker)

    scorer = adapter.load_scorer(
        kind="subject_pose",
        backend="scrfd_mediapipe",
        model_path=model_root,
        local_files_only=True,
    )

    assert scorer.model_root == model_root.resolve()
    assert session_calls == [
        (str(scrfd_path.resolve()), ["CPUExecutionProvider"]),
    ]
    assert landmarker_calls == [landmarker_path.resolve()]


def test_scrfd_decoder_uses_mock_onnxruntime_outputs_deterministically(
    tmp_path: Path,
) -> None:
    score_outputs = []
    bbox_outputs = []
    landmark_outputs = []
    for stride in (8, 16, 32):
        count = (32 // stride) * (32 // stride)
        score_outputs.append(np.zeros((count, 1), dtype=np.float32))
        bbox_outputs.append(np.full((count, 4), 0.5, dtype=np.float32))
        landmark_outputs.append(np.zeros((count, 10), dtype=np.float32))
    score_outputs[0][5, 0] = 0.95
    score_outputs[0][10, 0] = 0.85
    bbox_outputs[0][10] = 1.0

    class FakeSession:
        def get_inputs(self) -> list[object]:
            return [SimpleNamespace(shape=[1, 3, 32, 32], name="input.1")]

        def run(
            self,
            outputs: object,
            inputs: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            assert outputs is None
            assert inputs["input.1"].shape == (1, 3, 32, 32)
            return [*score_outputs, *bbox_outputs, *landmark_outputs]

    class FakeOnnxRuntime:
        @staticmethod
        def InferenceSession(path: str, *, providers: list[str]) -> FakeSession:
            assert path == str(tmp_path / "scrfd.onnx")
            assert providers == ["CPUExecutionProvider"]
            return FakeSession()

    detector = adapter._ScrfdDetector(tmp_path / "scrfd.onnx", FakeOnnxRuntime)

    detections = detector.detect(Image.new("RGB", (32, 32), "white"))

    assert len(detections) == 2
    selected = adapter._select_face(detections)
    assert selected is not None
    assert selected.confidence == pytest.approx(0.85)
    assert selected.bbox_xyxy == pytest.approx((8.0, 8.0, 24.0, 24.0))


def test_load_scorer_fails_when_exact_local_model_is_missing(tmp_path: Path) -> None:
    model_root = tmp_path / "face-models"
    model_root.mkdir()
    (model_root / "scrfd_10g_bnkps.onnx").write_bytes(b"onnx")

    with pytest.raises(FileNotFoundError):
        adapter.load_scorer(
            kind="subject_pose",
            backend="scrfd_mediapipe",
            model_path=model_root,
            local_files_only=True,
        )
