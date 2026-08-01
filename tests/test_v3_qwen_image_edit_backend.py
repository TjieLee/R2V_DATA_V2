from __future__ import annotations

import sys
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.v3.config import RemoveConfig
from r2v_data_v2.v3.qwen_image_edit_backend import (
    QwenImageEditRemovalBackend,
    build_background_removal_prompt,
    pad_image_to_multiple,
)


class _Generator:
    def __init__(self, *, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


def _install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    events: list[object],
    *,
    active: tuple[str, ...] = ("object_remover",),
    load_error: Exception | None = None,
    output: object | None = None,
    pipeline_vae_scale_factor: int = 8,
    use_peft_backend: bool = True,
) -> type:
    torch = types.ModuleType("torch")
    torch.bfloat16 = object()
    torch.float16 = object()
    torch.float32 = object()
    torch.Generator = _Generator
    torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: events.append("empty_cache"),
    )


    class FakePipeline:
        vae_scale_factor = pipeline_vae_scale_factor
        def __init__(self) -> None:
            self.active = active

        @classmethod
        def from_pretrained(
            cls,
            path: str,
            *,
            torch_dtype: object,
            local_files_only: bool,
        ) -> FakePipeline:
            events.append(
                ("from_pretrained", Path(path), torch_dtype, local_files_only)
            )
            return cls()

        def set_progress_bar_config(self, *, disable: bool) -> None:
            events.append(("progress", disable))

        def to(self, device: str) -> FakePipeline:
            events.append(("to", device))
            return self

        def load_lora_weights(
            self,
            pretrained_model_name_or_path: str,
            *,
            weight_name: str,
            adapter_name: str,
        ) -> None:
            events.append(
                (
                    "load_lora",
                    Path(pretrained_model_name_or_path),
                    weight_name,
                    adapter_name,
                )
            )
            if load_error is not None:
                raise load_error

        def set_adapters(self, adapter_names: str) -> None:
            events.append(("activate", adapter_names))

        def get_active_adapters(self) -> tuple[str, ...]:
            events.append("query_active")
            return self.active

        def __call__(
            self,
            *,
            image: list[Image.Image],
            height: int,
            width: int,
            prompt: str,
            generator: _Generator,
            true_cfg_scale: float,
            negative_prompt: str,
            num_inference_steps: int,
            guidance_scale: float,
            num_images_per_prompt: int,
        ) -> object:
            events.append(
                (
                    "inference",
                    image,
                    height,
                    width,
                    prompt,
                    generator,
                    true_cfg_scale,
                    negative_prompt,
                    num_inference_steps,
                    guidance_scale,
                    num_images_per_prompt,
                )
            )
            result = (
                Image.new("RGB", image[0].size, (20, 30, 40))
                if output is None
                else output
            )
            return SimpleNamespace(images=[result])

    diffusers = types.ModuleType("diffusers")
    diffusers.QwenImageEditPlusPipeline = FakePipeline
    diffusers_utils = types.ModuleType("diffusers.utils")
    diffusers_utils.USE_PEFT_BACKEND = use_peft_backend
    diffusers.utils = diffusers_utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils)
    return FakePipeline


def _config(
    tmp_path: Path,
    *,
    adapter_path: Path | None = None,
    adapter_weight_name: str | None = None,
) -> RemoveConfig:
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    adapter = adapter_path
    if adapter is None:
        adapter = tmp_path / "adapter.safetensors"
        adapter.write_bytes(b"lora")
    return RemoveConfig(
        base_model_path=base,
        adapter_path=adapter,
        adapter_weight_name=adapter_weight_name,
    )


def _remove(
    backend: QwenImageEditRemovalBackend,
    *,
    image: Image.Image | None = None,
) -> Image.Image:
    if image is None:
        image = Image.new("RGB", (3, 2), (1, 2, 3))
    return backend.remove(
        image=image,
        removal_phrases=["red car"],
        background_phrase="empty street",
        prompt="remove red car",
        seed=17,
    )


def test_pad_aligned_640x480_returns_rgb_copy_without_padding() -> None:
    source = Image.new("RGB", (640, 480), (10, 20, 30))

    padded, crop_box = pad_image_to_multiple(source, multiple=16)

    assert padded.size == source.size
    assert padded.mode == "RGB"
    assert padded is not source
    assert crop_box == (0, 0, 640, 480)
    assert np.array_equal(np.asarray(padded), np.asarray(source))


def test_pad_1280x590_to_model_multiple() -> None:
    source = Image.new("RGB", (1280, 590), (10, 20, 30))

    padded, crop_box = pad_image_to_multiple(source, multiple=16)

    assert padded.size == (1280, 592)
    assert crop_box == (0, 1, 1280, 591)


def test_pad_uses_edge_pixels_and_preserves_original_region() -> None:
    source_pixels = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ],
        dtype=np.uint8,
    )
    source = Image.fromarray(source_pixels, mode="RGB")

    padded, crop_box = pad_image_to_multiple(source, multiple=4)
    padded_pixels = np.asarray(padded)

    assert padded.size == (4, 4)
    assert crop_box == (0, 1, 3, 3)
    assert np.array_equal(padded_pixels[1:3, 0:3], source_pixels)
    assert np.array_equal(padded_pixels[0], padded_pixels[1])
    assert np.array_equal(padded_pixels[-1], padded_pixels[-2])
    assert np.array_equal(padded_pixels[:, -1], padded_pixels[:, -2])


def test_pad_distributes_odd_padding_as_symmetrically_as_possible() -> None:
    source = Image.new("RGB", (2, 4), (1, 2, 3))

    padded, crop_box = pad_image_to_multiple(source, multiple=5)

    left, top, right_edge, bottom_edge = crop_box
    right = padded.width - right_edge
    bottom = padded.height - bottom_edge
    assert padded.size == (5, 5)
    assert crop_box == (1, 0, 3, 4)
    assert abs(left - right) <= 1
    assert abs(top - bottom) <= 1


@pytest.mark.parametrize("multiple", [True, False, 1.5, "16"])
def test_pad_rejects_non_integer_multiple(multiple: object) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        pad_image_to_multiple(Image.new("RGB", (2, 2)), multiple=multiple)


@pytest.mark.parametrize("multiple", [0, -1])
def test_pad_rejects_non_positive_multiple(multiple: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        pad_image_to_multiple(Image.new("RGB", (2, 2)), multiple=multiple)


def test_pad_requires_pil_image() -> None:
    with pytest.raises(TypeError, match="PIL image"):
        pad_image_to_multiple("not-an-image", multiple=16)


def test_constructor_is_lazy_and_exposes_read_only_diagnostics(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    backend = QwenImageEditRemovalBackend(config)

    assert backend.base_model_path == config.base_model_path
    assert backend.adapter_path == config.adapter_path
    assert backend.adapter_weight_name is None
    assert backend.active_adapter_name is None
    assert backend._pipeline is None


def test_real_backend_requires_adapter_path(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), adapter_path=None)
    backend = QwenImageEditRemovalBackend(config)

    with pytest.raises(
        ValueError,
        match="Object-Remover LoRA adapter_path is required",
    ):
        _remove(backend)


def test_missing_adapter_path_fails_before_base_inference(tmp_path: Path) -> None:
    backend = QwenImageEditRemovalBackend(
        _config(tmp_path, adapter_path=tmp_path / "missing")
    )

    with pytest.raises(FileNotFoundError, match="adapter path does not exist"):
        _remove(backend)


def test_missing_base_model_fails_closed(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"lora")
    backend = QwenImageEditRemovalBackend(
        RemoveConfig(
            base_model_path=tmp_path / "missing-base",
            adapter_path=adapter,
        )
    )

    with pytest.raises(FileNotFoundError, match="base model path does not exist"):
        _remove(backend)


def test_missing_peft_backend_fails_before_loading_base_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(
        monkeypatch,
        events,
        use_peft_backend=False,
    )
    backend = QwenImageEditRemovalBackend(_config(tmp_path))

    with pytest.raises(
        RuntimeError,
        match="PEFT backend is required for the Object-Remover LoRA",
    ):
        _remove(backend)

    assert not any(
        isinstance(event, tuple) and event[0] == "from_pretrained"
        for event in events
    )


def test_single_file_lora_uses_parent_and_weight_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events)
    config = _config(tmp_path)
    assert config.adapter_path is not None
    backend = QwenImageEditRemovalBackend(config)

    result = _remove(backend)

    load = next(event for event in events if isinstance(event, tuple) and event[0] == "load_lora")
    assert load == (
        "load_lora",
        config.adapter_path.parent.resolve(),
        config.adapter_path.name,
        "object_remover",
    )
    assert isinstance(result, Image.Image)
    assert backend.adapter_weight_name == config.adapter_path.name
    assert backend.active_adapter_name == "object_remover"


def test_directory_lora_uses_the_unique_verified_weight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events)
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"lora")
    backend = QwenImageEditRemovalBackend(
        _config(tmp_path, adapter_path=directory)
    )

    _remove(backend)

    assert (
        "load_lora",
        directory.resolve(),
        "adapter_model.safetensors",
        "object_remover",
    ) in events


def test_configured_directory_weight_must_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events)
    directory = tmp_path / "adapter"
    directory.mkdir()
    backend = QwenImageEditRemovalBackend(
        _config(
            tmp_path,
            adapter_path=directory,
            adapter_weight_name="missing.safetensors",
        )
    )

    with pytest.raises(FileNotFoundError, match="configured.*weight"):
        _remove(backend)
    assert not any(
        isinstance(event, tuple) and event[0] == "from_pretrained"
        for event in events
    )


def test_configured_directory_weight_is_passed_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events)
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "chosen.safetensors").write_bytes(b"lora")
    (directory / "other.safetensors").write_bytes(b"lora")
    backend = QwenImageEditRemovalBackend(
        _config(
            tmp_path,
            adapter_path=directory,
            adapter_weight_name="chosen.safetensors",
        )
    )

    _remove(backend)

    assert (
        "load_lora",
        directory.resolve(),
        "chosen.safetensors",
        "object_remover",
    ) in events


def test_ambiguous_directory_weights_fail_without_guessing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events)
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "one.safetensors").write_bytes(b"1")
    (directory / "two.safetensors").write_bytes(b"2")
    backend = QwenImageEditRemovalBackend(
        _config(tmp_path, adapter_path=directory)
    )

    with pytest.raises(ValueError, match="adapter_weight_name"):
        _remove(backend)
    assert not any(
        isinstance(event, tuple) and event[0] == "from_pretrained"
        for event in events
    )


def test_unsupported_adapter_format_reports_required_calling_convention(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "adapter"
    directory.mkdir()
    (directory / "model_index.json").write_text("{}", encoding="utf-8")
    backend = QwenImageEditRemovalBackend(
        _config(tmp_path, adapter_path=directory)
    )

    with pytest.raises(RuntimeError, match="load_lora_weights"):
        _remove(backend)


def test_lora_load_failure_never_runs_base_only_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(
        monkeypatch,
        events,
        load_error=RuntimeError("bad lora"),
    )
    backend = QwenImageEditRemovalBackend(_config(tmp_path))

    with pytest.raises(RuntimeError, match="bad lora"):
        _remove(backend)

    names = [event[0] for event in events if isinstance(event, tuple)]
    assert "from_pretrained" in names
    assert "load_lora" in names
    assert "inference" not in names


def test_cached_load_failure_does_not_retry_with_base_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(
        monkeypatch,
        events,
        load_error=RuntimeError("bad lora"),
    )
    backend = QwenImageEditRemovalBackend(_config(tmp_path))

    with pytest.raises(RuntimeError, match="bad lora"):
        _remove(backend)
    with pytest.raises(RuntimeError, match="previously failed"):
        _remove(backend)

    assert sum(
        isinstance(event, tuple) and event[0] == "from_pretrained"
        for event in events
    ) == 1
    assert not any(
        isinstance(event, tuple) and event[0] == "inference"
        for event in events
    )


@pytest.mark.parametrize("active", [(), ("other_adapter",)])
def test_inactive_or_wrong_adapter_never_runs_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active: tuple[str, ...],
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events, active=active)
    backend = QwenImageEditRemovalBackend(_config(tmp_path))

    with pytest.raises(RuntimeError, match="not active"):
        _remove(backend)

    assert not any(
        isinstance(event, tuple) and event[0] == "inference"
        for event in events
    )


def test_load_activate_and_inference_order_and_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events)
    config = replace(
        _config(tmp_path),
        true_cfg_scale=4.5,
        guidance_scale=1.25,
        negative_prompt="none",
        num_inference_steps=23,
        device="cuda:1",
    )
    backend = QwenImageEditRemovalBackend(config)

    _remove(backend)

    names = [
        event if isinstance(event, str) else event[0]
        for event in events
    ]
    assert names.index("from_pretrained") < names.index("load_lora")
    assert names.index("load_lora") < names.index("activate")
    assert names.index("activate") < names.index("query_active")
    assert names.index("query_active") < names.index("inference")
    from_pretrained = events[names.index("from_pretrained")]
    assert from_pretrained[3] is True
    inference = events[names.index("inference")]
    assert inference[1][0].size == (16, 16)
    assert inference[2:4] == (16, 16)
    assert inference[5].device == "cuda:1"
    assert inference[5].seed == 17
    assert inference[6:] == (4.5, "none", 23, 1.25, 1)


def test_backend_passes_padded_image_and_explicit_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events)
    backend = QwenImageEditRemovalBackend(_config(tmp_path))
    source = Image.new("RGB", (1280, 590), (1, 2, 3))

    result = _remove(backend, image=source)

    inference = next(
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "inference"
    )
    padded_image = inference[1][0]
    assert padded_image.size == (1280, 592)
    assert padded_image.mode == "RGB"
    assert inference[2:4] == (592, 1280)
    assert result.size == source.size
    assert source.size == (1280, 590)


def test_backend_aligned_input_keeps_requested_dimensions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events)
    backend = QwenImageEditRemovalBackend(_config(tmp_path))
    source = Image.new("RGB", (640, 480), (1, 2, 3))

    result = _remove(backend, image=source)

    inference = next(
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "inference"
    )
    assert inference[1][0].size == source.size
    assert inference[2:4] == (480, 640)
    assert result.size == source.size


def test_backend_crops_padded_output_back_to_original_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_pixels = np.zeros((16, 16, 3), dtype=np.uint8)
    output_pixels[..., 0] = np.arange(16, dtype=np.uint8)[None, :]
    output_pixels[..., 1] = np.arange(16, dtype=np.uint8)[:, None]
    padded_output = Image.fromarray(output_pixels, mode="RGB")
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events, output=padded_output)
    backend = QwenImageEditRemovalBackend(_config(tmp_path))

    result = _remove(backend)

    assert result.size == (3, 2)
    assert result.mode == "RGB"
    assert np.array_equal(
        np.asarray(result),
        output_pixels[7:9, 6:9],
    )


def test_wrong_padded_output_size_fails_without_resize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(
        monkeypatch,
        events,
        output=Image.new("RGB", (1024, 1024)),
    )
    backend = QwenImageEditRemovalBackend(_config(tmp_path))

    def fail_if_resized(
        self: Image.Image,
        size: tuple[int, int],
        *args: object,
        **kwargs: object,
    ) -> Image.Image:
        raise AssertionError(f"resize must not be called: {self.size} -> {size}")

    monkeypatch.setattr(Image.Image, "resize", fail_if_resized)
    with pytest.raises(
        RuntimeError,
        match=(
            r"expected=\(16, 16\), actual=\(1024, 1024\)"
        ),
    ):
        _remove(backend)


def test_pipeline_signature_must_support_height_and_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    pipeline_type = _install_fake_runtime(monkeypatch, events)

    def incompatible_call(
        self: object,
        *,
        image: list[Image.Image],
        prompt: str,
        generator: _Generator,
        true_cfg_scale: float,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        num_images_per_prompt: int,
    ) -> object:
        raise AssertionError("inference must not run")

    monkeypatch.setattr(pipeline_type, "__call__", incompatible_call)
    backend = QwenImageEditRemovalBackend(_config(tmp_path))

    with pytest.raises(RuntimeError, match=r"height.*width"):
        _remove(backend)

    assert not any(
        isinstance(event, tuple) and event[0] == "inference"
        for event in events
    )


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_dtype_mapping_uses_installed_torch_attribute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: str,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events)
    backend = QwenImageEditRemovalBackend(
        replace(_config(tmp_path), dtype=dtype)
    )

    _remove(backend)

    loaded_dtype = next(
        event[2]
        for event in events
        if isinstance(event, tuple) and event[0] == "from_pretrained"
    )
    assert loaded_dtype is getattr(sys.modules["torch"], dtype)


def test_non_pil_pipeline_output_fails_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events, output="not-an-image")
    backend = QwenImageEditRemovalBackend(_config(tmp_path))

    with pytest.raises(TypeError, match="did not return a PIL image"):
        _remove(backend)


def test_close_releases_pipeline_and_cuda_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    _install_fake_runtime(monkeypatch, events)
    backend = QwenImageEditRemovalBackend(_config(tmp_path))
    _remove(backend)

    backend.close()

    assert backend.active_adapter_name is None
    assert backend._pipeline is None
    assert "empty_cache" in events


def test_prompt_is_deterministic_and_contains_all_safety_contracts() -> None:
    prompt = build_background_removal_prompt(
        removal_phrases=["red car", "driver"],
        background_phrase="empty stone street",
    )

    assert prompt == build_background_removal_prompt(
        removal_phrases=["red car", "driver"],
        background_phrase="empty stone street",
    )
    for phrase in ("red car", "driver", "empty stone street"):
        assert phrase in prompt
    normalized = prompt.casefold()
    assert "do not reconstruct" in normalized
    assert "do not add a person" in normalized
    assert "do not alter unrelated image regions" in normalized
