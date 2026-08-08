from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from PIL import Image


def _load_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoProcessor,
        Dinov2Model,
        Siglip2Model,
    )

    return torch, AutoImageProcessor, AutoProcessor, Dinov2Model, Siglip2Model


def _runtime(torch: Any) -> tuple[str, Any]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def _dtype_name(dtype: object) -> str:
    return str(dtype).removeprefix("torch.")


def _move_inputs(inputs: object, *, device: str, dtype: Any) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        try:
            inputs = dict(inputs)
        except (TypeError, ValueError) as exc:
            raise TypeError("processor output must be a tensor mapping") from exc
    moved: dict[str, Any] = {}
    for name, value in inputs.items():
        if not hasattr(value, "to"):
            raise TypeError(f"processor output {name!r} is not tensor-like")
        is_floating = getattr(value, "is_floating_point", None)
        if callable(is_floating) and is_floating():
            moved[name] = value.to(device=device, dtype=dtype)
        else:
            moved[name] = value.to(device=device)
    return moved


def _finite_vector(tensor: Any) -> list[float]:
    vector = tensor.detach().float().cpu().reshape(-1).tolist()
    if not vector or not all(math.isfinite(float(value)) for value in vector):
        raise ValueError("embedding vector must be finite and non-empty")
    return [float(value) for value in vector]


class _TransformersEmbeddingScorer:
    def __init__(
        self,
        *,
        backend: str,
        processor: object,
        model: object,
        torch: Any,
        device: str,
        dtype: Any,
    ) -> None:
        self.backend = backend
        self.processor = processor
        self.model = model
        self.torch = torch
        self.device = device
        self.dtype = dtype

    def eval(self) -> _TransformersEmbeddingScorer:
        eval_method = getattr(self.model, "eval", None)
        if not callable(eval_method):
            raise TypeError("embedding model must define eval()")
        eval_method()
        return self

    def _inputs(self, image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
        rgb = image.convert("RGB")
        processor = self.processor
        if not callable(processor):
            raise TypeError("embedding processor must be callable")
        values = processor(images=rgb, return_tensors="pt")
        return rgb, _move_inputs(values, device=self.device, dtype=self.dtype)

    def embed(self, image: Image.Image) -> dict[str, object]:
        original_width, original_height = image.size
        rgb, inputs = self._inputs(image)
        if rgb.size != (original_width, original_height):
            raise RuntimeError("RGB conversion changed source image dimensions")
        if self.backend == "dinov2":
            outputs = self.model(**inputs)
            hidden_state = getattr(outputs, "last_hidden_state", None)
            if hidden_state is None:
                raise RuntimeError("DINOv2 output is missing last_hidden_state")
            vector = _finite_vector(hidden_state[:, 0])
            embedding_source = "cls_token"
        elif self.backend == "siglip2":
            get_image_features = getattr(self.model, "get_image_features", None)
            if not callable(get_image_features):
                raise RuntimeError("SigLIP2 model is missing get_image_features()")
            vector = _finite_vector(get_image_features(**inputs))
            embedding_source = "get_image_features"
        else:
            raise ValueError(f"unsupported embedding backend: {self.backend}")
        return {
            "embedding": vector,
            "raw_metrics": {
                "embedding_source": embedding_source,
                "embedding_dimension": len(vector),
                "input_width": original_width,
                "input_height": original_height,
                "device": self.device,
                "dtype": _dtype_name(self.dtype),
            },
        }


def load_scorer(
    *,
    kind: str,
    backend: str,
    model_path: Path,
    local_files_only: bool,
) -> _TransformersEmbeddingScorer:
    if kind != "embedding":
        raise ValueError("transformers reference adapter only supports embedding")
    if backend not in {"dinov2", "siglip2"}:
        raise ValueError(f"unsupported embedding backend: {backend}")
    if local_files_only is not True:
        raise ValueError("transformers reference adapter requires local_files_only")
    resolved_model_path = Path(model_path).expanduser().resolve(strict=True)
    torch, auto_image_processor, auto_processor, dinov2_model, siglip2_model = (
        _load_runtime_dependencies()
    )
    device, dtype = _runtime(torch)
    common = {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    if backend == "dinov2":
        processor = auto_image_processor.from_pretrained(
            resolved_model_path,
            **common,
        )
        model = dinov2_model.from_pretrained(
            resolved_model_path,
            torch_dtype=dtype,
            **common,
        )
    else:
        processor = auto_processor.from_pretrained(
            resolved_model_path,
            **common,
        )
        model = siglip2_model.from_pretrained(
            resolved_model_path,
            torch_dtype=dtype,
            **common,
        )
    model = model.to(device)
    scorer = _TransformersEmbeddingScorer(
        backend=backend,
        processor=processor,
        model=model,
        torch=torch,
        device=device,
        dtype=dtype,
    )
    return scorer.eval()
