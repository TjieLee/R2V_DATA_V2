from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.v3.config import (
    SUBJECT_ATTRIBUTE_GME_MODEL_NAME,
    SubjectAttributeGmeConfig,
)
from r2v_data_v2.v3.subject_attribute_gme import (
    GmeAttributeWorkerConfig,
    PersistentGmeAttributeScreener,
    TEXT_TO_IMAGE_INSTRUCTION,
    build_negative_descriptions,
    build_positive_description,
    relative_margin_result,
)
from tools import run_v3_gme_attribute_worker as worker


def _request(image_path: Path, request_id: str = "request-1") -> dict[str, object]:
    return {
        "schema_version": 1,
        "type": "score",
        "request_id": request_id,
        "input_image_path": str(image_path),
        "instruction": TEXT_TO_IMAGE_INSTRUCTION,
        "positive_text": build_positive_description(
            phrase="black upper garment",
            attribute_type="upper_clothing",
        ),
        "negative_texts": build_negative_descriptions("upper_clothing"),
    }


class _EmbeddingModel:
    def __init__(self) -> None:
        self.inputs: list[list[dict[str, str]]] = []

    def encode(self, inputs, **kwargs):
        self.inputs.append(inputs)
        assert kwargs == {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        if "image" in inputs[0]:
            return np.array([[1.0, 0.0]], dtype=np.float32)
        return np.array(
            [[0.8, 0.6], [0.6, 0.8], [0.0, 1.0], [-1.0, 0.0], [0.2, 0.98]],
            dtype=np.float32,
        )


def test_relative_margin_has_no_absolute_score_gate() -> None:
    below = relative_margin_result(
        positive_score=-0.20,
        negative_scores={"owner": -0.10, "background": -0.30},
        min_margin=0.0,
    )
    equal = relative_margin_result(
        positive_score=-0.20,
        negative_scores={"owner": -0.20, "background": -0.30},
        min_margin=0.0,
    )

    assert below.margin == pytest.approx(-0.10)
    assert below.passed is False
    assert equal.margin == pytest.approx(0.0)
    assert equal.passed is True
    assert set(SubjectAttributeGmeConfig.__dataclass_fields__) == {
        "enabled",
        "backend",
        "python_executable",
        "model_path",
        "model_name",
        "screen_mode",
        "min_margin",
        "timeout_seconds",
    }


def test_gme_queries_are_english_and_type_specific_negatives_are_conservative() -> None:
    positive = build_positive_description(
        phrase="black hair",
        attribute_type="hair",
    )
    hair_negatives = build_negative_descriptions("hair")
    glasses_negatives = build_negative_descriptions("glasses")

    assert positive.isascii()
    assert all(value.isascii() for value in hair_negatives.values())
    assert "hair_fragment" in hair_negatives
    assert set(glasses_negatives) == {"owner_body", "background", "generic_fragment"}
    with pytest.raises(ValueError, match="English"):
        build_positive_description(phrase="黑色头发", attribute_type="hair")


def test_worker_scores_one_image_with_finite_official_style_embeddings(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "crop.png"
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(image_path)
    model = _EmbeddingModel()

    response = worker._score_request(_request(image_path), model=model)

    assert response["positive_score"] == pytest.approx(0.8)
    assert all(
        np.isfinite(value)
        for value in response["negative_scores"].values()
    )
    assert len(model.inputs) == 2
    assert all(item["prompt"] == TEXT_TO_IMAGE_INSTRUCTION for item in model.inputs[0])
    assert model.inputs[1] == [{"image": str(image_path)}]


def test_worker_loads_model_once_across_multiple_requests(tmp_path: Path) -> None:
    image_path = tmp_path / "crop.png"
    Image.new("RGB", (8, 8), "white").save(image_path)
    source = io.StringIO(
        "\n".join(
            json.dumps(_request(image_path, request_id))
            for request_id in ("request-1", "request-2")
        )
        + "\n"
    )
    output = io.StringIO()
    loads: list[tuple[Path, str]] = []
    model = _EmbeddingModel()

    def load_model(model_path: Path, *, device: str):
        loads.append((model_path, device))
        return model

    result = worker.serve(
        argparse.Namespace(
            model_path=tmp_path / "model",
            model_name=SUBJECT_ATTRIBUTE_GME_MODEL_NAME,
            device="cuda:0",
        ),
        input_stream=source,
        output_stream=output,
        model_loader=load_model,
    )

    responses = [json.loads(line) for line in output.getvalue().splitlines()]
    assert result == 0
    assert loads == [(tmp_path / "model", "cuda:0")]
    assert [response["request_id"] for response in responses[1:]] == [
        "request-1",
        "request-2",
    ]


def test_persistent_worker_isolates_one_physical_gpu_and_uses_cuda_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "venv" / "bin" / "python"
    executable.parent.mkdir(parents=True)
    executable.touch()
    model_path = tmp_path / "models" / "gme"
    model_path.mkdir(parents=True)
    captured: dict[str, object] = {}

    class _FakeProcess:
        stdin = io.StringIO()
        stdout = io.StringIO()

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout):
            return 0

        def kill(self):
            return None

    def popen(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return _FakeProcess()

    config = SubjectAttributeGmeConfig(
        enabled=True,
        python_executable=executable,
        model_path=model_path,
    )
    screener = PersistentGmeAttributeScreener(
        config,
        GmeAttributeWorkerConfig(
            python_executable=executable,
            model_path=model_path,
            model_name=SUBJECT_ATTRIBUTE_GME_MODEL_NAME,
            cuda_visible_devices="7",
            timeout_seconds=10,
            temporary_root=tmp_path / "run" / "gme_tmp",
            stderr_log_path=tmp_path / "run" / "gme.stderr.log",
        ),
        allowed_root=tmp_path,
    )
    monkeypatch.setattr("subprocess.Popen", popen)
    monkeypatch.setattr(
        screener,
        "_read_response",
        lambda: {"schema_version": 1, "status": "ok", "type": "ready"},
    )

    screener.start()
    environment = captured["environment"]
    command = captured["command"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert screener.starts == 1
    screener._terminate()


def test_worker_rejects_nonfinite_embeddings(tmp_path: Path) -> None:
    image_path = tmp_path / "crop.png"
    Image.new("RGB", (8, 8), "white").save(image_path)

    class _NonfiniteModel:
        def encode(self, inputs, **kwargs):
            if "image" in inputs[0]:
                return np.array([[1.0, 0.0]])
            return np.full((5, 2), np.nan)

    with pytest.raises(ValueError, match="finite embedding"):
        worker._score_request(_request(image_path), model=_NonfiniteModel())
