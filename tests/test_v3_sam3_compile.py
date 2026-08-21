from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from r2v_data_v2.v3.config import Sam3Config
from r2v_data_v2.v3.sam3_backend import Sam3SegmentationBackend


def _output() -> dict[str, object]:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    return {
        "out_binary_masks": mask[None, ...],
        "out_probs": np.array([0.9]),
        "out_obj_ids": np.array([1]),
    }


class _Predictor:
    def __init__(self, *, fail_first_propagation: bool = False) -> None:
        self.fail_first_propagation = fail_first_propagation
        self.requests: list[tuple[str, object]] = []
        self.shutdown_called = False
        self._session = 0

    def handle_request(self, request: dict[str, object]) -> dict[str, object]:
        request_type = str(request["type"])
        self.requests.append((request_type, request.get("frame_index")))
        if request_type == "start_session":
            self._session += 1
            return {"session_id": f"session-{self._session}"}
        if request_type == "close_session":
            return {"status": "ok"}
        return {"frame_index": request["frame_index"], "outputs": _output()}

    def handle_stream_request(
        self,
        request: dict[str, object],
    ) -> Any:
        direction = str(request["propagation_direction"])
        self.requests.append(("propagate_in_video", direction))
        if self.fail_first_propagation:
            self.fail_first_propagation = False
            raise RuntimeError("compile execution failed")
        slots = range(6, 10) if direction == "forward" else range(4, -1, -1)
        for slot in slots:
            yield {"frame_index": slot, "outputs": _output()}

    def shutdown(self) -> None:
        self.shutdown_called = True


def _frames(tmp_path: Path) -> list[Path]:
    frames = tmp_path / "frames"
    frames.mkdir()
    paths = []
    for slot in range(10):
        path = frames / f"{slot:02d}.jpg"
        path.write_bytes(b"frame")
        paths.append(path)
    return paths


def _config(tmp_path: Path) -> Sam3Config:
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"checkpoint")
    return Sam3Config(model_path=checkpoint)


def _track(backend: Sam3SegmentationBackend, frame_paths: list[Path]) -> None:
    result = backend.track(
        frame_paths=frame_paths,
        entity_id="e1",
        reference_type="subject",
        grounding_prompt="the person",
        entity_phrase="person",
    )
    assert result.status == "ready"


def test_eager_builder_arguments_remain_unchanged(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def builder(**kwargs: object) -> _Predictor:
        calls.append(kwargs)
        return _Predictor()

    config = _config(tmp_path)
    backend = Sam3SegmentationBackend(config, builder=builder)

    backend._load_predictor()

    assert calls == [{"checkpoint_path": str(config.model_path.resolve())}]
    assert backend.performance_counters()["sam3_compile_requested"] is False


def test_compile_uses_official_builder_option_and_reports_effective(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def builder(**kwargs: object) -> _Predictor:
        calls.append(kwargs)
        return _Predictor()

    backend = Sam3SegmentationBackend(
        _config(tmp_path),
        builder=builder,
        compile_enabled=True,
    )
    _track(backend, _frames(tmp_path))

    assert calls[0]["compile"] is True
    metrics = backend.performance_counters()
    assert metrics["sam3_compile_requested"] is True
    assert metrics["sam3_compile_effective"] is True
    assert metrics["sam3_compile_fallbacks"] == 0
    assert metrics["sam3_compile_failure_reason"] is None


def test_compile_construction_failure_rebuilds_eager_once(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def builder(**kwargs: object) -> _Predictor:
        calls.append(kwargs)
        if kwargs.get("compile") is True:
            raise RuntimeError("compile construction failed")
        return _Predictor()

    backend = Sam3SegmentationBackend(
        _config(tmp_path),
        builder=builder,
        compile_enabled=True,
    )
    _track(backend, _frames(tmp_path))

    assert calls[0]["compile"] is True
    assert "compile" not in calls[1]
    metrics = backend.performance_counters()
    assert metrics["sam3_compile_effective"] is False
    assert metrics["sam3_compile_fallbacks"] == 1
    assert metrics["sam3_compile_failure_reason"] == (
        "RuntimeError:compile construction failed"
    )


def test_first_compiled_execution_failure_replays_exactly_once_eager(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []
    predictors: list[_Predictor] = []

    def builder(**kwargs: object) -> _Predictor:
        calls.append(kwargs)
        predictor = _Predictor(fail_first_propagation=kwargs.get("compile") is True)
        predictors.append(predictor)
        return predictor

    backend = Sam3SegmentationBackend(
        _config(tmp_path),
        builder=builder,
        compile_enabled=True,
    )
    _track(backend, _frames(tmp_path))

    assert len(calls) == 2
    assert calls[0]["compile"] is True
    assert "compile" not in calls[1]
    assert predictors[0].shutdown_called is True
    eager_propagations = [
        value for name, value in predictors[1].requests if name == "propagate_in_video"
    ]
    assert eager_propagations == ["forward", "backward"]
    metrics = backend.performance_counters()
    assert metrics["sam3_compile_effective"] is False
    assert metrics["sam3_compile_fallbacks"] == 1
    assert metrics["sam3_compile_failure_reason"] == (
        "RuntimeError:compile execution failed"
    )
