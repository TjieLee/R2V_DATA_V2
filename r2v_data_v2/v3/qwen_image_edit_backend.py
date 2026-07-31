from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from r2v_data_v2.v3.config import RemoveConfig

_OBJECT_REMOVER_ADAPTER = "object_remover"
_SUPPORTED_WEIGHT_SUFFIXES = {".safetensors", ".bin"}


class BackgroundRemovalBackend(Protocol):
    def remove(
        self,
        *,
        image: Image.Image,
        removal_phrases: list[str],
        background_phrase: str,
        prompt: str,
        seed: int,
    ) -> Image.Image: ...


def build_background_removal_prompt(
    *,
    removal_phrases: list[str],
    background_phrase: str,
) -> str:
    phrases = [phrase.strip() for phrase in removal_phrases if phrase.strip()]
    if not phrases:
        raise ValueError("background removal requires at least one removal phrase")
    background = background_phrase.strip()
    if not background:
        raise ValueError("background removal requires a background phrase")
    return (
        "Remove the specified foreground entities: "
        + "; ".join(phrases)
        + ". Replace those regions only with a seamless continuation of the "
        f"surrounding background: {background}. Preserve geometry, perspective, "
        "depth, texture, lighting, color, shadows, and camera characteristics. "
        "Do not reconstruct or replace the removed entities. Do not add a person, "
        "animal, vehicle, product, text, sign, or any other salient object. "
        "Do not alter unrelated image regions."
    )


def _supports_parameter(callable_object: Any, name: str) -> bool:
    try:
        parameters = inspect.signature(callable_object).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"cannot inspect installed API signature for {callable_object!r}"
        ) from exc
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _require_parameters(
    callable_object: Any,
    names: tuple[str, ...],
    *,
    operation: str,
) -> None:
    missing = [
        name
        for name in names
        if not _supports_parameter(callable_object, name)
    ]
    if missing:
        signature = inspect.signature(callable_object)
        raise RuntimeError(
            f"installed diffusers does not support {operation} parameters "
            f"{missing}; actual signature is {signature}"
        )


class QwenImageEditRemovalBackend:
    def __init__(self, config: RemoveConfig) -> None:
        self.config = config
        self._pipeline: Any | None = None
        self._torch: Any | None = None
        self._load_error: BaseException | None = None
        self._adapter_weight_name = config.adapter_weight_name
        self._active_adapter_name: str | None = None

    @property
    def base_model_path(self) -> Path:
        return self.config.base_model_path

    @property
    def adapter_path(self) -> Path | None:
        return self.config.adapter_path

    @property
    def adapter_weight_name(self) -> str | None:
        return self._adapter_weight_name

    @property
    def active_adapter_name(self) -> str | None:
        return self._active_adapter_name

    def _resolve_adapter_weight(self, adapter_path: Path) -> tuple[Path, str]:
        if adapter_path.is_file():
            if adapter_path.suffix.casefold() != ".safetensors":
                raise RuntimeError(
                    "unsupported Object-Remover LoRA adapter format: a single "
                    "adapter file must be a .safetensors file loadable with "
                    "load_lora_weights(parent, weight_name=filename)"
                )
            if self.config.adapter_weight_name not in {
                None,
                adapter_path.name,
            }:
                raise ValueError(
                    "remove.adapter_weight_name does not match the configured "
                    "single-file Object-Remover LoRA"
                )
            return adapter_path.parent, adapter_path.name

        if not adapter_path.is_dir():
            raise FileNotFoundError(
                f"Object-Remover LoRA adapter path does not exist: {adapter_path}"
            )
        configured = self.config.adapter_weight_name
        if configured is not None:
            candidate = (adapter_path / configured).resolve(strict=False)
            try:
                candidate.relative_to(adapter_path.resolve())
            except ValueError as exc:
                raise ValueError(
                    "remove.adapter_weight_name must remain inside adapter_path"
                ) from exc
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"configured Object-Remover LoRA weight does not exist: {candidate}"
                )
            if candidate.suffix.casefold() not in _SUPPORTED_WEIGHT_SUFFIXES:
                raise RuntimeError(
                    "unsupported Object-Remover LoRA weight format for "
                    f"remove.adapter_weight_name: {candidate.suffix or '<none>'}"
                )
            return candidate.parent, candidate.name

        candidates = sorted(
            path
            for path in adapter_path.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in _SUPPORTED_WEIGHT_SUFFIXES
        )
        if not candidates:
            marker = adapter_path / "adapter_config.json"
            detail = (
                "adapter_config.json exists but no compatible weight file was found"
                if marker.is_file()
                else "no adapter_config.json or compatible LoRA weight was found"
            )
            raise RuntimeError(
                "unsupported Object-Remover LoRA adapter format: "
                f"{detail}; use a verified Diffusers/PEFT LoRA and its required "
                "load_lora_weights calling convention"
            )
        if len(candidates) > 1:
            names = [
                candidate.relative_to(adapter_path).as_posix()
                for candidate in candidates
            ]
            raise ValueError(
                "multiple Object-Remover LoRA weights are present; set "
                f"remove.adapter_weight_name explicitly: {names}"
            )
        return candidates[0].parent, candidates[0].name

    @staticmethod
    def _active_adapter_names(pipeline: Any) -> tuple[str, ...]:
        getter = getattr(pipeline, "get_active_adapters", None)
        if callable(getter):
            value = getter()
        else:
            value = getattr(pipeline, "active_adapters", None)
            if callable(value):
                value = value()
        if value is None:
            raise RuntimeError(
                "installed diffusers cannot report active LoRA adapters"
            )
        if isinstance(value, str):
            return (value,)
        if isinstance(value, dict):
            return tuple(str(name) for name in value)
        try:
            return tuple(str(name) for name in value)
        except TypeError as exc:
            raise RuntimeError(
                "installed diffusers returned an unsupported active adapter format"
            ) from exc

    def _load(self) -> None:
        adapter_path = self.config.adapter_path
        if adapter_path is None:
            raise ValueError("Object-Remover LoRA adapter_path is required")
        base_path = self.config.base_model_path.expanduser().resolve(strict=False)
        resolved_adapter = adapter_path.expanduser().resolve(strict=False)
        if not base_path.is_dir():
            raise FileNotFoundError(
                f"Qwen-Image-Edit-2511 base model path does not exist: {base_path}"
            )
        if not resolved_adapter.exists():
            raise FileNotFoundError(
                "Object-Remover LoRA adapter path does not exist: "
                f"{resolved_adapter}"
            )
        adapter_directory, weight_name = self._resolve_adapter_weight(
            resolved_adapter
        )

        import torch
        from diffusers import QwenImageEditPlusPipeline

        dtype = getattr(torch, self.config.dtype, None)
        if dtype is None:
            raise RuntimeError(
                f"installed torch does not provide dtype {self.config.dtype}"
            )
        from_pretrained = QwenImageEditPlusPipeline.from_pretrained
        _require_parameters(
            from_pretrained,
            ("torch_dtype", "local_files_only"),
            operation="QwenImageEditPlusPipeline.from_pretrained",
        )
        pipeline = from_pretrained(
            str(base_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
        try:
            progress = getattr(pipeline, "set_progress_bar_config", None)
            if not callable(progress):
                raise TypeError(
                    "installed QwenImageEditPlusPipeline cannot disable progress bars"
                )
            _require_parameters(
                progress,
                ("disable",),
                operation="set_progress_bar_config",
            )
            progress(disable=True)

            to_method = getattr(pipeline, "to", None)
            if not callable(to_method):
                raise TypeError(
                    "installed QwenImageEditPlusPipeline has no device transfer API"
                )
            pipeline = to_method(self.config.device)

            load_lora = getattr(pipeline, "load_lora_weights", None)
            if not callable(load_lora):
                raise TypeError(
                    "installed diffusers does not support load_lora_weights for "
                    "QwenImageEditPlusPipeline"
                )
            _require_parameters(
                load_lora,
                ("weight_name", "adapter_name"),
                operation="load_lora_weights",
            )
            load_lora(
                str(adapter_directory),
                weight_name=weight_name,
                adapter_name=_OBJECT_REMOVER_ADAPTER,
            )

            activate = getattr(pipeline, "set_adapters", None)
            if not callable(activate):
                raise TypeError(
                    "installed diffusers cannot activate the Object-Remover LoRA; "
                    "set_adapters is unavailable"
                )
            activate_signature = inspect.signature(activate)
            if not activate_signature.parameters:
                raise RuntimeError(
                    "installed diffusers set_adapters API cannot accept an adapter name"
                )
            activate(_OBJECT_REMOVER_ADAPTER)
            active_names = self._active_adapter_names(pipeline)
            if _OBJECT_REMOVER_ADAPTER not in active_names:
                raise RuntimeError(
                    "Object-Remover LoRA was loaded but is not active; "
                    f"active adapters: {list(active_names)}"
                )
        except Exception:
            del pipeline
            raise

        self._pipeline = pipeline
        self._torch = torch
        self._adapter_weight_name = weight_name
        self._active_adapter_name = _OBJECT_REMOVER_ADAPTER

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(
                "Object-Remover LoRA backend previously failed to load"
            ) from self._load_error
        try:
            self._load()
        except Exception as exc:
            self._load_error = exc
            raise

    def remove(
        self,
        *,
        image: Image.Image,
        removal_phrases: list[str],
        background_phrase: str,
        prompt: str,
        seed: int,
    ) -> Image.Image:
        if not isinstance(image, Image.Image):
            raise TypeError("Qwen removal input image must be a PIL image")
        if not removal_phrases or not all(
            isinstance(phrase, str) and phrase.strip()
            for phrase in removal_phrases
        ):
            raise ValueError("Qwen removal phrases must be non-empty strings")
        if not isinstance(background_phrase, str) or not background_phrase.strip():
            raise ValueError("Qwen background phrase must be non-empty")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Qwen removal prompt must be non-empty")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("Qwen removal seed must be a non-negative integer")

        self._ensure_loaded()
        if (
            self._pipeline is None
            or self._torch is None
            or self._active_adapter_name != _OBJECT_REMOVER_ADAPTER
        ):
            raise RuntimeError("Object-Remover LoRA backend is not active")

        generator = self._torch.Generator(
            device=self.config.device
        ).manual_seed(seed)
        parameters = {
            "image": [image.convert("RGB")],
            "prompt": prompt,
            "generator": generator,
            "true_cfg_scale": self.config.true_cfg_scale,
            "negative_prompt": self.config.negative_prompt,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "num_images_per_prompt": 1,
        }
        _require_parameters(
            self._pipeline.__call__,
            tuple(parameters),
            operation="QwenImageEditPlusPipeline.__call__",
        )
        result = self._pipeline(**parameters)
        images = getattr(result, "images", None)
        if (
            not isinstance(images, (list, tuple))
            or not images
            or not isinstance(images[0], Image.Image)
        ):
            raise TypeError("Qwen removal backend did not return a PIL image")
        return images[0]

    def close(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._active_adapter_name = None
        if pipeline is not None:
            del pipeline
        torch = self._torch
        self._torch = None
        if (
            torch is not None
            and hasattr(torch, "cuda")
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
