from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from tools.reference_filter_adapters.transformers_embedding import (
    r2v_reference_filter_adapter as adapter,
)


class FakeTensor:
    def __init__(self, values: object, *, floating: bool = True) -> None:
        self.values = np.asarray(values)
        self.floating = floating
        self.moves: list[dict[str, object]] = []

    def is_floating_point(self) -> bool:
        return self.floating

    def to(self, **kwargs: object) -> FakeTensor:
        self.moves.append(dict(kwargs))
        return self

    def __getitem__(self, key: object) -> FakeTensor:
        return FakeTensor(self.values[key], floating=self.floating)

    def detach(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def reshape(self, *shape: int) -> FakeTensor:
        return FakeTensor(self.values.reshape(*shape), floating=self.floating)

    def tolist(self) -> list[float]:
        return self.values.tolist()


class FakeProcessor:
    load_calls: list[tuple[Path, dict[str, object]]]

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict[str, FakeTensor]:
        self.calls.append(dict(kwargs))
        image = kwargs["images"]
        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"
        return {
            "pixel_values": FakeTensor([[[[0.25]]]]),
            "pixel_attention_mask": FakeTensor([[1]], floating=False),
        }


class FakeProcessorFactory:
    load_calls: ClassVar[list[tuple[Path, dict[str, object]]]] = []
    instances: ClassVar[list[FakeProcessor]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        **kwargs: object,
    ) -> FakeProcessor:
        cls.load_calls.append((model_path, dict(kwargs)))
        instance = FakeProcessor()
        cls.instances.append(instance)
        return instance


class FakeModel:
    def __init__(self, *, embedding: object) -> None:
        self.embedding = embedding
        self.to_calls: list[str] = []
        self.eval_calls = 0
        self.forward_calls: list[dict[str, object]] = []
        self.image_feature_calls: list[dict[str, object]] = []

    def to(self, device: str) -> FakeModel:
        self.to_calls.append(device)
        return self

    def eval(self) -> FakeModel:
        self.eval_calls += 1
        return self

    def __call__(self, **kwargs: object) -> object:
        self.forward_calls.append(dict(kwargs))
        return SimpleNamespace(last_hidden_state=FakeTensor(self.embedding))

    def get_image_features(self, **kwargs: object) -> FakeTensor:
        self.image_feature_calls.append(dict(kwargs))
        return FakeTensor(self.embedding)


class FakeDinov2ModelFactory:
    load_calls: ClassVar[list[tuple[Path, dict[str, object]]]] = []
    instances: ClassVar[list[FakeModel]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        **kwargs: object,
    ) -> FakeModel:
        cls.load_calls.append((model_path, dict(kwargs)))
        instance = FakeModel(embedding=[[[1.0, 2.0, 3.0], [9.0, 9.0, 9.0]]])
        cls.instances.append(instance)
        return instance


class FakeSiglip2ModelFactory:
    load_calls: ClassVar[list[tuple[Path, dict[str, object]]]] = []
    instances: ClassVar[list[FakeModel]] = []

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path,
        **kwargs: object,
    ) -> FakeModel:
        cls.load_calls.append((model_path, dict(kwargs)))
        instance = FakeModel(embedding=[[4.0, 5.0, 6.0, 7.0]])
        cls.instances.append(instance)
        return instance


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    cuda = FakeCuda()
    float16 = "torch.float16"
    float32 = "torch.float32"


@pytest.fixture(autouse=True)
def reset_factories() -> None:
    for factory in (
        FakeProcessorFactory,
        FakeDinov2ModelFactory,
        FakeSiglip2ModelFactory,
    ):
        factory.load_calls.clear()
        factory.instances.clear()


@pytest.fixture
def fake_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        adapter,
        "_load_runtime_dependencies",
        lambda: (
            FakeTorch,
            FakeProcessorFactory,
            FakeProcessorFactory,
            FakeDinov2ModelFactory,
            FakeSiglip2ModelFactory,
        ),
    )


def _model_path(tmp_path: Path) -> Path:
    path = tmp_path / "local-model"
    path.mkdir()
    return path


def _assert_local_load(
    calls: list[tuple[Path, dict[str, object]]],
    model_path: Path,
) -> None:
    assert len(calls) == 1
    assert calls[0][0] == model_path
    assert calls[0][1]["local_files_only"] is True
    assert calls[0][1]["trust_remote_code"] is False


def test_dinov2_uses_fixed_cls_embedding_and_loads_once(
    tmp_path: Path,
    fake_dependencies: None,
) -> None:
    model_path = _model_path(tmp_path)
    scorer = adapter.load_scorer(
        kind="embedding",
        backend="dinov2",
        model_path=model_path,
        local_files_only=True,
    )
    source = Image.new("RGBA", (19, 23), (10, 20, 30, 127))
    source_bytes = source.tobytes()

    first = scorer.embed(source)
    second = scorer.embed(source)

    assert first["embedding"] == [1.0, 2.0, 3.0]
    assert second["embedding"] == [1.0, 2.0, 3.0]
    assert all(np.isfinite(first["embedding"]))
    assert first["raw_metrics"] == {
        "embedding_source": "cls_token",
        "embedding_dimension": 3,
        "input_width": 19,
        "input_height": 23,
        "device": "cpu",
        "dtype": "float32",
    }
    _assert_local_load(FakeProcessorFactory.load_calls, model_path)
    _assert_local_load(FakeDinov2ModelFactory.load_calls, model_path)
    assert FakeDinov2ModelFactory.instances[0].eval_calls == 1
    assert len(FakeDinov2ModelFactory.instances[0].forward_calls) == 2
    assert source.mode == "RGBA"
    assert source.tobytes() == source_bytes


def test_siglip2_uses_image_features_without_text_and_loads_once(
    tmp_path: Path,
    fake_dependencies: None,
) -> None:
    model_path = _model_path(tmp_path)
    scorer = adapter.load_scorer(
        kind="embedding",
        backend="siglip2",
        model_path=model_path,
        local_files_only=True,
    )
    source = Image.new("RGB", (31, 17), (1, 2, 3))

    result = scorer.embed(source)
    scorer.embed(source)

    assert result["embedding"] == [4.0, 5.0, 6.0, 7.0]
    assert result["raw_metrics"]["embedding_source"] == "get_image_features"
    assert result["raw_metrics"]["embedding_dimension"] == 4
    _assert_local_load(FakeProcessorFactory.load_calls, model_path)
    _assert_local_load(FakeSiglip2ModelFactory.load_calls, model_path)
    model = FakeSiglip2ModelFactory.instances[0]
    assert model.eval_calls == 1
    assert len(model.image_feature_calls) == 2
    assert not model.forward_calls
    assert all("input_ids" not in call for call in model.image_feature_calls)
    assert all("text" not in call for call in FakeProcessorFactory.instances[0].calls)


@pytest.mark.parametrize(
    ("kind", "backend", "message"),
    [
        ("quality", "dinov2", "only supports embedding"),
        ("embedding", "onealign", "unsupported embedding backend"),
    ],
)
def test_adapter_rejects_unsupported_modes_before_loading(
    tmp_path: Path,
    fake_dependencies: None,
    kind: str,
    backend: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        adapter.load_scorer(
            kind=kind,
            backend=backend,
            model_path=_model_path(tmp_path),
            local_files_only=True,
        )
    assert not FakeProcessorFactory.load_calls
    assert not FakeDinov2ModelFactory.load_calls
    assert not FakeSiglip2ModelFactory.load_calls


def test_adapter_rejects_network_enabled_loading(
    tmp_path: Path,
    fake_dependencies: None,
) -> None:
    with pytest.raises(ValueError, match="requires local_files_only"):
        adapter.load_scorer(
            kind="embedding",
            backend="dinov2",
            model_path=_model_path(tmp_path),
            local_files_only=False,
        )
    source = inspect.getsource(adapter)
    assert "OpenAI" not in source
    assert "Qwen" not in source
    assert "text=" not in source


def test_nonfinite_embedding_fails_closed(
    tmp_path: Path,
    fake_dependencies: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scorer = adapter.load_scorer(
        kind="embedding",
        backend="dinov2",
        model_path=_model_path(tmp_path),
        local_files_only=True,
    )
    model = FakeDinov2ModelFactory.instances[0]
    monkeypatch.setattr(model, "embedding", [[[float("nan")]]])
    with pytest.raises(ValueError, match="finite and non-empty"):
        scorer.embed(Image.new("RGB", (8, 8)))
