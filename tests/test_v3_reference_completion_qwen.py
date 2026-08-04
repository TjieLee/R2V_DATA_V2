from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import r2v_data_v2.v3.reference_completion_benchmark as completion_module
import r2v_data_v2.v3.reference_completion_qwen as qwen_module
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.reference_completion_benchmark import (
    QwenReferenceCompletionJudge,
    ReferenceCompletionReview,
)
from r2v_data_v2.v3.reference_completion_qwen import (
    DEFAULT_QWEN_COMPLETION_MODE,
    DEFAULT_QWEN_COMPOSITING_MODE,
    DEFAULT_QWEN_LOCALIZED_GROUP_PROMPT,
    DEFAULT_QWEN_LOCALIZED_NEGATIVE_PROMPT,
    DEFAULT_QWEN_LOCALIZED_OBJECT_PROMPT,
    DEFAULT_QWEN_LOCALIZED_SUBJECT_PROMPT,
    DEFAULT_QWEN_MODEL_PATH,
    DEFAULT_QWEN_SEEDS,
    DEFAULT_QWEN_SUBJECT_COMPLETION_NEGATIVE_PROMPT,
    DEFAULT_QWEN_SUBJECT_COMPLETION_PROMPT,
    QWEN_COMPLETION_BACKEND,
    QWEN_LOCALIZED_JUDGE_SYSTEM_PROMPT,
    QWEN_LOCALIZED_PROMPT_ZH_SHORT,
    QWEN_WHOLE_CANVAS_JUDGE_ADDENDUM,
    QwenCompletionCompositingMode,
    QwenImageEdit2511CompletionConfig,
    QwenImageEdit2511ReferenceCompletionBackend,
    QwenLocalizedReferenceCompletionJudge,
    run_qwen_reference_completion_benchmark,
)
from tools.run_v3_reference_completion_qwen import _parser


@dataclass(frozen=True)
class _Environment:
    data_root: Path
    benchmark_base: Path
    output_root: Path
    manifest: Path
    source: Path
    completion_mask: Path
    model_path: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_image() -> Image.Image:
    pixels = np.zeros((10, 12, 4), dtype=np.uint8)
    pixels[:, :, :3] = (9, 13, 17)
    pixels[2:7, 3:9, :3] = (193, 79, 41)
    pixels[2:7, 3:9, 3] = 255
    return Image.fromarray(pixels)


def _completion_mask() -> Image.Image:
    pixels = np.zeros((15, 12), dtype=np.uint8)
    pixels[7:15, 4:8] = 255
    return Image.fromarray(pixels)


def _gradient(size: tuple[int, int], *, offset: int = 0) -> Image.Image:
    width, height = size
    yy, xx = np.indices((height, width))
    pixels = np.stack(
        (
            (xx * 17 + yy * 3 + 13 + offset) % 256,
            (yy * 29 + xx * 5 + 31 + offset) % 256,
            ((xx + yy) * 11 + 47 + offset) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    return Image.fromarray(pixels)


def _write_manifest(
    environment: _Environment,
    *,
    reference_type: str = "subject",
    completion_mask_path: Path | None | object = ...,
    entity_phrase: str = "a person in a red coat",
    completion_start_ratio: float = 0.5,
    completion_sides: list[str] | None = None,
) -> None:
    mask_path = (
        environment.completion_mask
        if completion_mask_path is ...
        else completion_mask_path
    )
    payload = {
        "sample_id": "sample-1",
        "clip_uid": "clip-1",
        "entity_id": "e1",
        "reference_type": reference_type,
        "entity_phrase": entity_phrase,
        "source_rgba_path": str(environment.source),
        "context_rgb_path": None,
        "completion_mask_path": str(mask_path) if mask_path is not None else None,
        "completion_sides": completion_sides or ["bottom"],
        "completion_start_ratio": completion_start_ratio,
    }
    environment.manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")


@pytest.fixture
def environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Environment:
    data_root = (tmp_path / "workspace" / "data").resolve()
    input_root = data_root / "inputs"
    benchmark_base = data_root / "reference_completion_qwen_benchmarks"
    model_path = data_root / "models" / "Qwen-Image-Edit-2511"
    input_root.mkdir(parents=True)
    benchmark_base.mkdir(parents=True)
    model_path.mkdir(parents=True)
    environment = _Environment(
        data_root=data_root,
        benchmark_base=benchmark_base,
        output_root=benchmark_base / "unit-test",
        manifest=data_root / "completion.jsonl",
        source=input_root / "entity.png",
        completion_mask=input_root / "completion-mask.png",
        model_path=model_path,
    )
    _source_image().save(environment.source, format="PNG")
    _completion_mask().save(environment.completion_mask, format="PNG")
    _write_manifest(environment)
    monkeypatch.setattr(completion_module, "ALLOWED_INPUT_ROOT", data_root)
    monkeypatch.setattr(qwen_module, "ALLOWED_QWEN_BENCHMARK_ROOT", benchmark_base)
    return environment


def _config(
    environment: _Environment,
    **updates: object,
) -> QwenImageEdit2511CompletionConfig:
    values: dict[str, object] = {
        "model_path": environment.model_path,
        "canvas_expand_ratio": 0.5,
        "lateral_padding_ratio": 0.2,
        "model_min_side": 64,
        "model_multiple": 8,
        "mode": "whole_canvas",
    }
    values.update(updates)
    return QwenImageEdit2511CompletionConfig(**values)


def _review_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "verdict": "accept",
        "visible_source_preserved": True,
        "same_entity_continued": True,
        "identity_preserved": True,
        "exactly_one_entity": True,
        "completion_plausible": True,
        "completion_useful": True,
        "no_occluder_reconstructed": True,
        "no_new_salient_entity": True,
        "boundary_clean": True,
        "reference_usable": True,
        "reason": "The same person is completed cleanly.",
    }
    payload.update(updates)
    return payload


def _accept() -> ReferenceCompletionReview:
    return ReferenceCompletionReview.model_validate(_review_payload())


def _reject(**updates: object) -> ReferenceCompletionReview:
    values = {
        "verdict": "reject",
        "completion_useful": False,
        "reason": "The completion is not useful.",
    }
    values.update(updates)
    return ReferenceCompletionReview.model_validate(_review_payload(**values))


@dataclass
class _Backend:
    outputs: list[Image.Image | BaseException] = field(default_factory=list)
    calls: list[dict[str, object]] = field(default_factory=list)

    def complete(self, **kwargs: object) -> Image.Image:
        self.calls.append(dict(kwargs))
        if self.outputs:
            output = self.outputs.pop(0)
            if isinstance(output, BaseException):
                raise output
            return output
        input_rgb = kwargs["input_rgb"]
        seed = kwargs["seed"]
        assert isinstance(input_rgb, Image.Image)
        assert isinstance(seed, int)
        return _gradient(input_rgb.size, offset=seed)


@dataclass
class _Judge:
    outcomes: list[ReferenceCompletionReview | BaseException]
    calls: list[dict[str, object]] = field(default_factory=list)

    def review(self, **kwargs: object) -> ReferenceCompletionReview:
        self.calls.append(dict(kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _run(
    environment: _Environment,
    *,
    backend: _Backend | None = None,
    judge: _Judge | None = None,
    prompt: str = DEFAULT_QWEN_SUBJECT_COMPLETION_PROMPT,
    negative_prompt: str = DEFAULT_QWEN_SUBJECT_COMPLETION_NEGATIVE_PROMPT,
    seeds: tuple[int, ...] = (0, 17),
    compositing_mode: QwenCompletionCompositingMode = (
        DEFAULT_QWEN_COMPOSITING_MODE
    ),
) -> tuple[_Backend, _Judge]:
    selected_backend = backend or _Backend()
    selected_judge = judge or _Judge([_accept() for _ in seeds])
    run_qwen_reference_completion_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        config=_config(environment, mode=compositing_mode),
        backend=selected_backend,
        judge=selected_judge,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seeds=seeds,
    )
    return selected_backend, selected_judge


def _run_localized(
    environment: _Environment,
    *,
    backend: _Backend | None = None,
    judge: _Judge | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    seed: int = 0,
) -> tuple[_Backend, _Judge]:
    selected_backend = backend or _Backend()
    selected_judge = judge or _Judge([_accept()])
    run_qwen_reference_completion_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        config=_config(environment, mode="localized_raw"),
        backend=selected_backend,
        judge=selected_judge,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seeds=(seed,),
    )
    return selected_backend, selected_judge


def _result(environment: _Environment) -> dict[str, object]:
    return json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )


def test_default_configuration_and_prompts_are_local_and_english() -> None:
    config = QwenImageEdit2511CompletionConfig()

    assert config.model_path == DEFAULT_QWEN_MODEL_PATH
    assert config.local_files_only is True
    assert config.model_min_side == 1024
    assert config.model_multiple == 16
    assert config.mode == "localized_raw"
    assert config.compositing_mode is None
    assert DEFAULT_QWEN_COMPLETION_MODE == "localized_raw"
    assert DEFAULT_QWEN_COMPOSITING_MODE == "whole_canvas"
    assert DEFAULT_QWEN_SEEDS == (0,)
    assert DEFAULT_QWEN_LOCALIZED_NEGATIVE_PROMPT == " "
    assert "Complete the missing parts of the same single person" in (
        DEFAULT_QWEN_SUBJECT_COMPLETION_PROMPT
    )
    assert "second person" in DEFAULT_QWEN_SUBJECT_COMPLETION_NEGATIVE_PROMPT
    assert not any(
        "\u4e00" <= character <= "\u9fff"
        for character in DEFAULT_QWEN_SUBJECT_COMPLETION_PROMPT
    )


def test_config_requires_existing_local_model_directory(
    environment: _Environment,
) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        replace(_config(environment), model_path=environment.data_root / "missing").validate()


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"local_files_only": False}, "local_files_only"),
        ({"device": " "}, "device"),
        ({"dtype": "int8"}, "dtype"),
        ({"num_inference_steps": 0}, "positive integer"),
        ({"num_inference_steps": True}, "positive integer"),
        ({"true_cfg_scale": -0.1}, "non-negative"),
        ({"guidance_scale": float("inf")}, "finite"),
        ({"canvas_expand_ratio": 0.0}, "positive"),
        ({"lateral_padding_ratio": float("nan")}, "finite"),
        ({"mask_overlap_pixels": -1}, "non-negative integer"),
        ({"model_min_side": 0}, "positive integer"),
        ({"model_multiple": 0}, "positive integer"),
        ({"mode": "unknown"}, "mode"),
    ],
)
def test_config_rejects_invalid_values(
    environment: _Environment,
    updates: dict[str, object],
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        replace(_config(environment), **updates).validate()


def test_localized_config_ignores_legacy_geometry_settings(
    environment: _Environment,
) -> None:
    config = replace(
        _config(environment, mode="localized_raw"),
        canvas_expand_ratio=0.0,
        lateral_padding_ratio=float("nan"),
        mask_overlap_pixels=-1,
        model_min_side=0,
        model_multiple=0,
    )

    config.validate()


@dataclass
class _FakeGenerator:
    device: str
    seed: int | None = None

    def manual_seed(self, seed: int) -> _FakeGenerator:
        self.seed = seed
        return self


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    @staticmethod
    def is_available() -> bool:
        return True

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.progress_values: list[bool] = []
        self.devices: list[str] = []
        self.output: object | None = None

    def set_progress_bar_config(self, *, disable: bool) -> None:
        self.progress_values.append(disable)

    def to(self, device: str) -> _FakePipeline:
        self.devices.append(device)
        return self

    def __call__(
        self,
        *,
        image: list[Image.Image],
        prompt: str,
        negative_prompt: str,
        height: int,
        width: int,
        generator: _FakeGenerator,
        true_cfg_scale: float,
        guidance_scale: float,
        num_inference_steps: int,
        num_images_per_prompt: int,
    ) -> object:
        parameters = dict(locals())
        parameters.pop("self")
        self.calls.append(parameters)
        if self.output is not None:
            return self.output
        return SimpleNamespace(images=[_gradient((width, height))])


class _LocalizedFakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.output: object | None = None

    def __call__(
        self,
        *,
        image: list[Image.Image],
        prompt: str,
        negative_prompt: str,
        generator: _FakeGenerator,
        true_cfg_scale: float,
        guidance_scale: float,
        num_inference_steps: int,
        num_images_per_prompt: int,
    ) -> object:
        parameters = dict(locals())
        parameters.pop("self")
        self.calls.append(parameters)
        if self.output is not None:
            return self.output
        return SimpleNamespace(images=[_gradient(image[0].size)])


def _fake_torch() -> SimpleNamespace:
    return SimpleNamespace(
        float16="float16-value",
        bfloat16="bfloat16-value",
        float32="float32-value",
        Generator=_FakeGenerator,
        cuda=_FakeCuda(),
    )


def test_loader_uses_qwen_pipeline_local_path_dtype_and_no_lora(
    environment: _Environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[dict[str, object]] = []
    pipeline = _FakePipeline()

    class FakeQwenImageEditPlusPipeline:
        @classmethod
        def from_pretrained(
            cls,
            model_path: str,
            *,
            dtype: object,
            local_files_only: bool,
        ) -> _FakePipeline:
            loaded.append(
                {
                    "model_path": model_path,
                    "dtype": dtype,
                    "local_files_only": local_files_only,
                }
            )
            return pipeline

    torch = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(
            QwenImageEditPlusPipeline=FakeQwenImageEditPlusPipeline,
        ),
    )
    backend = QwenImageEdit2511ReferenceCompletionBackend(_config(environment))
    backend._ensure_loaded()

    assert loaded == [
        {
            "model_path": str(environment.model_path),
            "dtype": "bfloat16-value",
            "local_files_only": True,
        }
    ]
    assert pipeline.progress_values == [True]
    assert pipeline.devices == ["cuda"]
    assert not hasattr(pipeline, "load_lora_weights")


def test_loader_supports_legacy_torch_dtype_parameter(
    environment: _Environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: dict[str, object] = {}

    class LegacyQwenPipeline:
        @classmethod
        def from_pretrained(
            cls,
            model_path: str,
            *,
            torch_dtype: object,
            local_files_only: bool,
        ) -> _FakePipeline:
            loaded.update(
                model_path=model_path,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )
            return _FakePipeline()

    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(QwenImageEditPlusPipeline=LegacyQwenPipeline),
    )
    backend = QwenImageEdit2511ReferenceCompletionBackend(_config(environment))
    backend._ensure_loaded()

    assert loaded["torch_dtype"] == "bfloat16-value"
    assert loaded["local_files_only"] is True


def test_backend_sends_one_rgb_image_no_mask_and_all_inference_parameters(
    environment: _Environment,
) -> None:
    pipeline = _FakePipeline()
    torch = _fake_torch()
    config = replace(
        _config(environment),
        num_inference_steps=23,
        true_cfg_scale=3.5,
        guidance_scale=1.25,
    )
    backend = QwenImageEdit2511ReferenceCompletionBackend(
        config,
        pipeline=pipeline,
        torch_module=torch,
    )
    input_rgb = Image.new("RGB", (32, 48), "white")

    output = backend.complete(
        input_rgb=input_rgb,
        entity_phrase="a person",
        seed=17,
        prompt="Complete the person.",
        negative_prompt="extra person",
    )

    assert output.mode == "RGB"
    assert output.size == input_rgb.size
    assert len(pipeline.calls) == 1
    request = pipeline.calls[0]
    assert request["image"] == [input_rgb]
    assert "mask" not in request and "completion_mask" not in request
    assert request["height"] == 48 and request["width"] == 32
    assert request["prompt"] == "Complete the person."
    assert request["negative_prompt"] == "extra person"
    assert request["true_cfg_scale"] == 3.5
    assert request["guidance_scale"] == 1.25
    assert request["num_inference_steps"] == 23
    assert request["num_images_per_prompt"] == 1
    assert request["generator"].device == "cuda"
    assert request["generator"].seed == 17


def test_localized_backend_uses_official_like_call_and_preserves_raw_size(
    environment: _Environment,
) -> None:
    pipeline = _LocalizedFakePipeline()
    pipeline.output = SimpleNamespace(images=[Image.new("RGBA", (19, 23), "red")])
    config = replace(
        _config(environment, mode="localized_raw"),
        num_inference_steps=23,
        true_cfg_scale=3.5,
        guidance_scale=1.25,
    )
    backend = QwenImageEdit2511ReferenceCompletionBackend(
        config,
        pipeline=pipeline,
        torch_module=_fake_torch(),
    )
    input_rgb = Image.new("RGB", (32, 48), "white")

    output = backend.complete(
        input_rgb=input_rgb,
        entity_phrase="a ceramic vessel",
        seed=0,
        prompt=DEFAULT_QWEN_LOCALIZED_OBJECT_PROMPT,
        negative_prompt=" ",
    )

    assert output.mode == "RGB"
    assert output.size == (19, 23)
    assert len(pipeline.calls) == 1
    request = pipeline.calls[0]
    assert request["image"] == [input_rgb]
    assert request["prompt"] == DEFAULT_QWEN_LOCALIZED_OBJECT_PROMPT
    assert request["negative_prompt"] == " "
    assert request["true_cfg_scale"] == 3.5
    assert request["guidance_scale"] == 1.25
    assert request["num_inference_steps"] == 23
    assert request["num_images_per_prompt"] == 1
    assert request["generator"].device == "cuda"
    assert request["generator"].seed == 0
    assert {"height", "width", "mask", "completion_mask"}.isdisjoint(request)


def test_localized_backend_can_force_exact_production_input_size(
    environment: _Environment,
) -> None:
    pipeline = _FakePipeline()
    config = replace(
        _config(environment, mode="localized_raw"),
        force_input_size=True,
    )
    backend = QwenImageEdit2511ReferenceCompletionBackend(
        config,
        pipeline=pipeline,
        torch_module=_fake_torch(),
    )
    input_rgb = Image.new("RGB", (37, 53), "white")

    output = backend.complete(
        input_rgb=input_rgb,
        entity_phrase="a ceramic vessel",
        seed=0,
        prompt=DEFAULT_QWEN_LOCALIZED_OBJECT_PROMPT,
        negative_prompt=" ",
    )

    assert output.mode == "RGB"
    assert output.size == input_rgb.size
    request = pipeline.calls[0]
    assert request["height"] == 53
    assert request["width"] == 37
    assert request["image"] == [input_rgb]


@pytest.mark.parametrize(
    ("output", "error"),
    [
        (SimpleNamespace(images=None), "did not return"),
        (SimpleNamespace(images=[]), "did not return"),
        (SimpleNamespace(images=["not-an-image"]), "did not return"),
        (SimpleNamespace(images=[Image.new("RGBA", (32, 48))]), "RGB mode"),
        (SimpleNamespace(images=[Image.new("RGB", (31, 48))]), "dimensions"),
    ],
)
def test_backend_fails_closed_on_malformed_output(
    environment: _Environment,
    output: object,
    error: str,
) -> None:
    pipeline = _FakePipeline()
    pipeline.output = output
    backend = QwenImageEdit2511ReferenceCompletionBackend(
        _config(environment),
        pipeline=pipeline,
        torch_module=_fake_torch(),
    )
    with pytest.raises((TypeError, RuntimeError), match=error):
        backend.complete(
            input_rgb=Image.new("RGB", (32, 48), "white"),
            entity_phrase="a person",
            seed=0,
            prompt="Complete the person.",
            negative_prompt="extra person",
        )


def test_backend_close_releases_pipeline_and_cuda_cache(
    environment: _Environment,
) -> None:
    torch = _fake_torch()
    backend = QwenImageEdit2511ReferenceCompletionBackend(
        _config(environment),
        pipeline=_FakePipeline(),
        torch_module=torch,
    )

    backend.close()

    assert backend._pipeline is None
    assert backend._torch is None
    assert torch.cuda.empty_cache_calls == 1


@pytest.mark.parametrize(
    ("reference_type", "expected_prompt"),
    [
        ("subject", DEFAULT_QWEN_LOCALIZED_SUBJECT_PROMPT),
        ("object", DEFAULT_QWEN_LOCALIZED_OBJECT_PROMPT),
        ("group", DEFAULT_QWEN_LOCALIZED_GROUP_PROMPT),
    ],
)
def test_localized_raw_supports_each_reference_type_and_default_prompt(
    environment: _Environment,
    reference_type: str,
    expected_prompt: str,
) -> None:
    _write_manifest(
        environment,
        reference_type=reference_type,
        entity_phrase="distinct evidence phrase",
    )
    source_sha256 = _sha256(environment.source)

    backend, judge = _run_localized(environment)
    result = _result(environment)
    sample_root = environment.output_root / "sample-1"

    assert len(backend.calls) == 1
    assert backend.calls[0]["input_rgb"].size == _source_image().size
    assert backend.calls[0]["prompt"] == expected_prompt
    assert backend.calls[0]["negative_prompt"] == " "
    assert "distinct evidence phrase" not in backend.calls[0]["prompt"]
    assert judge.calls[0]["reference_type"] == reference_type
    assert set(judge.calls[0]) == {
        "source_rgba",
        "candidate_rgb",
        "entity_phrase",
        "reference_type",
    }
    assert result["mode"] == "localized_raw"
    assert result["reference_type"] == reference_type
    assert result["prompt"] == expected_prompt
    assert result["prompt_language"] == "en"
    assert result["negative_prompt"] == " "
    assert result["seed"] == 0
    assert result["input_path"] == "input_source_white.png"
    assert result["input_size"] == [12, 10]
    assert result["candidate_path"] == "candidate_qwen_localized_seed_0.png"
    assert result["output_size"] == [12, 10]
    assert result["status"] == "manual_review_pending"
    assert result["manual_review_required"] is True
    assert result["accepted_candidate"] is None
    assert result["localized_completion"] is True
    assert result["full_instance_reconstruction_requested"] is False
    assert result["canvas_expansion_used"] is False
    assert result["completion_mask_used"] is False
    assert result["visible_pixel_restore_used"] is False
    assert result["height_width_forced"] is False
    assert result["entity_phrase_appended_to_prompt"] is False
    assert result["source_sha256"] == source_sha256
    assert _sha256(environment.source) == source_sha256
    assert result["candidate_sha256"] == _sha256(
        sample_root / "candidate_qwen_localized_seed_0.png"
    )
    assert {
        "completion_mask_area_ratio",
        "editable_region_path",
        "source_offset_xy",
        "canvas_size",
        "compositing_mode",
    }.isdisjoint(result)
    assert {path.name for path in sample_root.iterdir()} == {
        "source_rgba.png",
        "input_source_white.png",
        "candidate_qwen_localized_seed_0.png",
        "review_qwen_localized_seed_0.json",
        "result.json",
    }


@pytest.mark.parametrize("reference_type", ["subject", "object", "group"])
def test_localized_prompt_override_applies_without_entity_phrase(
    environment: _Environment,
    reference_type: str,
) -> None:
    _write_manifest(
        environment,
        reference_type=reference_type,
        entity_phrase="never append this phrase",
    )

    backend, _ = _run_localized(
        environment,
        prompt=QWEN_LOCALIZED_PROMPT_ZH_SHORT,
    )
    result = _result(environment)

    assert backend.calls[0]["prompt"] == QWEN_LOCALIZED_PROMPT_ZH_SHORT
    assert "never append this phrase" not in backend.calls[0]["prompt"]
    assert result["prompt_language"] == "zh"


def test_localized_raw_ignores_legacy_completion_geometry(
    environment: _Environment,
) -> None:
    ignored_mask = environment.data_root.parent / "outside-mask-that-must-not-be-read.png"
    _write_manifest(
        environment,
        completion_mask_path=ignored_mask,
        completion_sides=["top", "left", "right"],
        completion_start_ratio=1.0,
    )

    backend, _ = _run_localized(environment)
    result = _result(environment)

    assert backend.calls[0]["input_rgb"].size == (12, 10)
    assert result["input_size"] == [12, 10]
    assert result["completion_mask_used"] is False
    assert not (environment.output_root / "sample-1" / "completion_mask.png").exists()
    assert not (environment.output_root / "sample-1" / "editable_region.png").exists()


def test_localized_raw_preserves_pipeline_output_size_without_restoration(
    environment: _Environment,
) -> None:
    class DifferentSizeBackend(_Backend):
        def complete(self, **kwargs: object) -> Image.Image:
            self.calls.append(dict(kwargs))
            return _gradient((7, 9), offset=91)

    _run_localized(environment, backend=DifferentSizeBackend())
    result = _result(environment)
    sample_root = environment.output_root / "sample-1"
    with Image.open(sample_root / "candidate_qwen_localized_seed_0.png") as opened:
        opened.load()
        assert opened.mode == "RGB"
        assert opened.size == (7, 9)
        assert np.array_equal(np.asarray(opened), np.asarray(_gradient((7, 9), offset=91)))

    assert result["input_size"] == [12, 10]
    assert result["output_size"] == [7, 9]
    assert result["visible_pixel_restore_used"] is False
    assert not list(sample_root.glob("visible_restored_candidate_*.png"))


@pytest.mark.parametrize(
    ("backend_mode", "reason"),
    [
        ("unchanged", "candidate_unchanged_from_input"),
        ("constant", "candidate_is_constant"),
    ],
)
def test_localized_raw_hard_rejects_unchanged_or_constant_output(
    environment: _Environment,
    backend_mode: str,
    reason: str,
) -> None:
    class HardFailureBackend(_Backend):
        def complete(self, **kwargs: object) -> Image.Image:
            self.calls.append(dict(kwargs))
            input_rgb = kwargs["input_rgb"]
            assert isinstance(input_rgb, Image.Image)
            if backend_mode == "unchanged":
                return input_rgb.copy()
            return Image.new("RGB", input_rgb.size, (77, 77, 77))

    judge = _Judge([_accept()])
    _run_localized(
        environment,
        backend=HardFailureBackend(),
        judge=judge,
    )
    result = _result(environment)
    checks = result["hard_check"]["checks"]

    assert result["status"] == "rejected"
    assert result["accepted_candidate"] is None
    assert result["judge_status"] == "not_run"
    assert reason in result["hard_check"]["reasons"]
    assert {
        "source_visible_pixels_exact",
        "outside_completion_mask_exact",
        "completion_mask_connected_to_visible_entity",
        "candidate_changed_inside_completion_mask",
    }.isdisjoint(checks)
    assert judge.calls == []
    summary = json.loads(
        (environment.output_root / "benchmark_summary.json").read_text()
    )
    assert summary["hard_check_rejection_counts"][reason] == 1


@pytest.mark.parametrize("reference_type", ["subject", "object", "group"])
def test_localized_judge_rejection_fails_closed_for_every_reference_type(
    environment: _Environment,
    reference_type: str,
) -> None:
    _write_manifest(environment, reference_type=reference_type)
    _run_localized(
        environment,
        judge=_Judge([_reject(exactly_one_entity=False)]),
    )
    result = _result(environment)

    assert result["status"] == "rejected"
    assert result["judge_status"] == "reviewed"
    assert result["judge_verdict"] == "reject"
    assert result["accepted_candidate"] is None


def test_localized_judge_sends_only_source_and_candidate_with_type_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = QwenLocalizedReferenceCompletionJudge(
        QwenServiceConfig(model="fake-local-judge"),
        client=SimpleNamespace(),
    )
    captured: list[list[dict[str, object]]] = []

    def fake_request(messages: list[dict[str, object]]) -> str:
        captured.append(messages)
        return json.dumps(_review_payload())

    monkeypatch.setattr(judge, "_request", fake_request)
    review = judge.review(
        source_rgba=_source_image(),
        candidate_rgb=_gradient((17, 19)),
        entity_phrase="a ceramic vessel",
        reference_type="object",
    )

    assert review.verdict == "accept"
    assert len(captured) == 1
    messages = captured[0]
    assert QWEN_LOCALIZED_JUDGE_SYSTEM_PROMPT in messages[0]["content"]
    assert "For an object" in messages[0]["content"]
    user_content = messages[1]["content"]
    labels = [item["text"] for item in user_content if item["type"] == "text"]
    assert labels[1:] == [
        "Image 1: source entity on white",
        "Image 2: localized raw candidate",
    ]
    assert len([item for item in user_content if item["type"] == "image_url"]) == 2
    assert not any("mask" in label.casefold() for label in labels)


def test_localized_summary_counts_reference_types_and_manual_review(
    environment: _Environment,
) -> None:
    base = json.loads(environment.manifest.read_text(encoding="utf-8"))
    records = []
    for index, reference_type in enumerate(("subject", "object", "group"), start=1):
        record = dict(base)
        record.update(
            sample_id=f"sample-{index}",
            entity_id=f"e{index}",
            reference_type=reference_type,
        )
        records.append(record)
    environment.manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    backend = _Backend()
    judge = _Judge([_accept(), _accept(), _accept()])
    stats = run_qwen_reference_completion_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        config=_config(environment, mode="localized_raw"),
        backend=backend,
        judge=judge,
    )
    summary = json.loads(
        (environment.output_root / "benchmark_summary.json").read_text()
    )

    assert stats.to_dict() == {"processed": 3, "accepted": 0, "rejected": 0}
    assert summary == {
        "backend": QWEN_COMPLETION_BACKEND,
        "mode": "localized_raw",
        "processed": 3,
        "reference_type_counts": {"group": 1, "object": 1, "subject": 1},
        "manual_review_pending": 3,
        "rejected": 0,
        "accepted": 0,
        "hard_check_rejection_counts": {},
        "judge_rejection_flag_counts": {},
    }


def test_localized_unknown_reference_type_fails_before_root_creation(
    environment: _Environment,
) -> None:
    payload = json.loads(environment.manifest.read_text(encoding="utf-8"))
    payload["reference_type"] = "unknown"
    environment.manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid completion manifest"):
        _run_localized(environment)

    assert not environment.output_root.exists()


@pytest.mark.parametrize(
    ("source_mode", "error"),
    [
        ("rgb", "RGBA PNG"),
        ("partial_alpha", "only 0 and 255"),
        ("empty_alpha", "foreground is empty"),
    ],
)
def test_localized_source_rgba_validation_fails_before_root_creation(
    environment: _Environment,
    source_mode: str,
    error: str,
) -> None:
    if source_mode == "rgb":
        Image.new("RGB", (12, 10), "white").save(environment.source, format="PNG")
    else:
        pixels = np.asarray(_source_image()).copy()
        pixels[:, :, 3] = 128 if source_mode == "partial_alpha" else 0
        Image.fromarray(pixels).save(environment.source, format="PNG")

    with pytest.raises(ValueError, match=error):
        _run_localized(environment)

    assert not environment.output_root.exists()


@pytest.mark.parametrize("reference_type", ["object", "group"])
def test_non_subject_reference_fails_before_root_creation(
    environment: _Environment,
    reference_type: str,
) -> None:
    _write_manifest(environment, reference_type=reference_type)
    backend = _Backend()
    judge = _Judge([_accept()])

    with pytest.raises(ValueError, match="only reference_type=subject"):
        _run(environment, backend=backend, judge=judge, seeds=(0,))

    assert not environment.output_root.exists()
    assert backend.calls == [] and judge.calls == []


def test_missing_explicit_mask_fails_before_root_creation(
    environment: _Environment,
) -> None:
    _write_manifest(environment, completion_mask_path=None)
    backend = _Backend()
    judge = _Judge([_accept()])

    with pytest.raises(ValueError, match="requires completion_mask_path"):
        _run(
            environment,
            backend=backend,
            judge=judge,
            seeds=(0,),
            compositing_mode="explicit_mask",
        )

    assert not environment.output_root.exists()
    assert backend.calls == [] and judge.calls == []


def test_invalid_explicit_mask_keeps_shared_fail_closed_validation(
    environment: _Environment,
) -> None:
    Image.new("L", (12, 15), 0).save(environment.completion_mask)
    backend = _Backend()
    judge = _Judge([_accept()])

    with pytest.raises(ValueError, match="empty"):
        _run(
            environment,
            backend=backend,
            judge=judge,
            seeds=(0,),
            compositing_mode="explicit_mask",
        )

    assert not environment.output_root.exists()
    assert backend.calls == [] and judge.calls == []


def test_qwen_run_uses_expanded_rgb_canvas_seed_order_and_full_prompt(
    environment: _Environment,
) -> None:
    backend, _ = _run(environment)

    assert [call["seed"] for call in backend.calls] == [0, 17]
    assert all(call["input_rgb"].mode == "RGB" for call in backend.calls)
    assert all(call["input_rgb"].size == (64, 80) for call in backend.calls)
    assert all("completion_mask" not in call for call in backend.calls)
    assert all(
        call["prompt"].endswith("Entity description: a person in a red coat.")
        for call in backend.calls
    )


def test_prompt_override_is_preserved_and_blank_prompts_are_rejected(
    environment: _Environment,
) -> None:
    backend, _ = _run(
        environment,
        prompt="Extend the same visible person only.",
        negative_prompt="duplicate person",
        seeds=(5,),
    )
    assert backend.calls[0]["prompt"] == (
        "Extend the same visible person only. "
        "Entity description: a person in a red coat."
    )
    assert backend.calls[0]["negative_prompt"] == "duplicate person"

    for value_name in ("prompt", "negative_prompt"):
        output_root = environment.benchmark_base / f"blank-{value_name}"
        kwargs = {
            "prompt": DEFAULT_QWEN_SUBJECT_COMPLETION_PROMPT,
            "negative_prompt": DEFAULT_QWEN_SUBJECT_COMPLETION_NEGATIVE_PROMPT,
        }
        kwargs[value_name] = "   "
        with pytest.raises(ValueError, match=value_name.replace("_", " ")):
            run_qwen_reference_completion_benchmark(
                manifest_path=environment.manifest,
                benchmark_root=output_root,
                config=_config(environment),
                backend=_Backend(),
                judge=_Judge([_accept()]),
                seeds=(0,),
                **kwargs,
            )
        assert not output_root.exists()


def test_whole_canvas_result_uses_qwen_fields_and_keeps_both_seeds(
    environment: _Environment,
) -> None:
    _run(environment)
    result = _result(environment)

    assert result["backend"] == QWEN_COMPLETION_BACKEND
    assert result["compositing_mode"] == "whole_canvas"
    assert result["completion_mask_mode"] is None
    assert result["completion_mask_source_path"] is None
    assert result["completion_mask_source_sha256"] is None
    assert result["editable_region_path"] == "editable_region.png"
    assert [attempt["candidate_id"] for attempt in result["attempts"]] == [
        "qwen_seed_0",
        "qwen_seed_17",
    ]
    assert [attempt["candidate_path"] for attempt in result["attempts"]] == [
        "candidate_qwen_seed_0.png",
        "candidate_qwen_seed_17.png",
    ]
    assert [attempt["review_path"] for attempt in result["attempts"]] == [
        "review_qwen_seed_0.json",
        "review_qwen_seed_17.json",
    ]
    assert result["accepted_candidate"]["candidate_id"] == "qwen_seed_0"
    assert all("fitting_degree" not in attempt for attempt in result["attempts"])
    assert all(
        attempt["compositing_mode"] == "whole_canvas"
        for attempt in result["attempts"]
    )
    assert "fitting_degree" not in result["accepted_candidate"]
    assert all(attempt["prompt"] for attempt in result["attempts"])
    assert all(attempt["negative_prompt"] for attempt in result["attempts"])


def test_whole_canvas_allows_null_mask_and_publishes_editable_region(
    environment: _Environment,
) -> None:
    _write_manifest(environment, completion_mask_path=None)
    _, judge = _run(environment, seeds=(0,))
    sample_root = environment.output_root / "sample-1"

    with Image.open(sample_root / "editable_region.png") as opened:
        opened.load()
        editable = np.asarray(opened)
        assert opened.mode == "L"
        assert opened.size == (12, 15)
    with Image.open(sample_root / "visible_mask.png") as opened:
        visible = np.asarray(opened) == 255

    assert set(np.unique(editable)) == {0, 255}
    assert np.all(editable[visible] == 0)
    assert np.all(editable[~visible] == 255)
    assert not (sample_root / "completion_mask.png").exists()
    assert np.array_equal(
        np.asarray(judge.calls[0]["completion_mask"]),
        editable,
    )
    assert judge.calls[0]["completion_region_label"] == (
        "editable region; this is not a model input mask"
    )
    assert judge.calls[0]["system_prompt_addendum"] == (
        QWEN_WHOLE_CANVAS_JUDGE_ADDENDUM
    )


def test_qwen_judge_renders_whole_canvas_region_label_and_instructions() -> None:
    judge = QwenReferenceCompletionJudge(
        QwenServiceConfig(model="fake-local-judge"),
        client=SimpleNamespace(),
    )
    canvas = Image.new("RGB", (12, 15), "white")
    editable = Image.new("L", canvas.size, 255)
    messages = judge._messages(
        source_rgba=_source_image(),
        completion_canvas=canvas,
        completion_mask=editable,
        candidate_rgb=_gradient(canvas.size),
        request_text="Review.",
        completion_region_label=(
            "editable region; this is not a model input mask"
        ),
        system_prompt_addendum=QWEN_WHOLE_CANVAS_JUDGE_ADDENDUM,
    )

    assert QWEN_WHOLE_CANVAS_JUDGE_ADDENDUM in messages[0]["content"]
    assert any(
        item
        == {
            "type": "text",
            "text": (
                "Image 3: editable region; this is not a model input mask"
            ),
        }
        for item in messages[1]["content"]
    )


def test_whole_canvas_keeps_qwen_pixels_outside_visible_entity(
    environment: _Environment,
) -> None:
    _write_manifest(environment, completion_mask_path=None)
    _run(environment, seeds=(0,))
    sample_root = environment.output_root / "sample-1"
    with Image.open(sample_root / "candidate_qwen_seed_0.png") as opened:
        candidate = np.asarray(opened.convert("RGB"))
    with Image.open(sample_root / "baseline_canvas.png") as opened:
        baseline = np.asarray(opened.convert("RGB"))
    with Image.open(sample_root / "visible_mask.png") as opened:
        visible = np.asarray(opened) == 255
    expected_generated = np.asarray(
        _gradient((64, 80)).resize((12, 15), Image.Resampling.LANCZOS)
    )

    assert np.array_equal(candidate[visible], baseline[visible])
    assert np.array_equal(candidate[~visible], expected_generated[~visible])
    assert np.any(candidate[~visible] != baseline[~visible])


def test_whole_canvas_ignores_completion_start_ratio(
    environment: _Environment,
) -> None:
    first = replace(
        environment,
        output_root=environment.benchmark_base / "ratio-zero",
    )
    _write_manifest(
        first,
        completion_mask_path=None,
        completion_start_ratio=0.0,
    )
    _run(first, seeds=(0,))
    first_region = (first.output_root / "sample-1" / "editable_region.png").read_bytes()

    second = replace(
        environment,
        output_root=environment.benchmark_base / "ratio-one",
    )
    _write_manifest(
        second,
        completion_mask_path=None,
        completion_start_ratio=1.0,
    )
    _run(second, seeds=(0,))
    second_region = (
        second.output_root / "sample-1" / "editable_region.png"
    ).read_bytes()

    assert first_region == second_region
    assert _result(second)["completion_start_ratio"] == 1.0


def test_explicit_mask_result_and_artifacts_remain_available(
    environment: _Environment,
) -> None:
    _run(environment, seeds=(0,), compositing_mode="explicit_mask")
    result = _result(environment)
    attempt = result["attempts"][0]
    sample_root = environment.output_root / "sample-1"

    assert result["compositing_mode"] == "explicit_mask"
    assert result["completion_mask_mode"] == "explicit"
    assert result["completion_mask_source_path"] == str(
        environment.completion_mask
    )
    assert result["completion_mask_source_sha256"] == _sha256(
        environment.completion_mask
    )
    assert result["editable_region_path"] is None
    assert attempt["compositing_mode"] == "explicit_mask"
    assert "outside_completion_mask_exact" in attempt["hard_check"]["checks"]
    assert (sample_root / "completion_mask.png").exists()
    assert not (sample_root / "editable_region.png").exists()


@pytest.mark.parametrize(
    ("backend_mode", "reason"),
    [
        ("unchanged", "candidate_unchanged_outside_visible_region"),
        ("constant", "generated_region_is_constant"),
    ],
)
def test_whole_canvas_hard_rejections_skip_judge(
    environment: _Environment,
    backend_mode: str,
    reason: str,
) -> None:
    class HardFailureBackend(_Backend):
        def complete(self, **kwargs: object) -> Image.Image:
            self.calls.append(dict(kwargs))
            input_rgb = kwargs["input_rgb"]
            assert isinstance(input_rgb, Image.Image)
            if backend_mode == "unchanged":
                return input_rgb.copy()
            return Image.new("RGB", input_rgb.size, (77, 77, 77))

    judge = _Judge([_accept()])
    _run(
        environment,
        backend=HardFailureBackend(),
        judge=judge,
        seeds=(0,),
    )
    attempt = _result(environment)["attempts"][0]

    assert attempt["judge_status"] == "not_run"
    assert reason in attempt["hard_check"]["reasons"]
    assert "outside_completion_mask_exact" not in attempt["hard_check"]["checks"]
    assert judge.calls == []


def test_whole_canvas_hard_checks_use_editable_region_fields(
    environment: _Environment,
) -> None:
    _run(environment, seeds=(0,))
    checks = _result(environment)["attempts"][0]["hard_check"]["checks"]

    assert checks["candidate_changed_outside_visible_region"] is True
    assert checks["generated_region_not_constant"] is True
    assert "outside_completion_mask_exact" not in checks
    assert "candidate_changed_inside_completion_mask" not in checks


def test_hard_compositing_restores_visible_and_outside_mask_pixels(
    environment: _Environment,
) -> None:
    source_hash = _sha256(environment.source)
    mask_hash = _sha256(environment.completion_mask)
    _run(environment, seeds=(0,), compositing_mode="explicit_mask")
    sample_root = environment.output_root / "sample-1"
    with Image.open(sample_root / "candidate_qwen_seed_0.png") as candidate_image:
        candidate = np.asarray(candidate_image.convert("RGB"))
    with Image.open(sample_root / "baseline_canvas.png") as baseline_image:
        baseline = np.asarray(baseline_image.convert("RGB"))
    with Image.open(sample_root / "visible_mask.png") as visible_image:
        visible = np.asarray(visible_image) == 255
    with Image.open(sample_root / "completion_mask.png") as mask_image:
        mask = np.asarray(mask_image) == 255

    assert np.array_equal(candidate[visible], baseline[visible])
    assert np.array_equal(candidate[~mask], baseline[~mask])
    assert np.any(candidate[mask] != baseline[mask])
    assert len(np.unique(candidate[mask].reshape(-1, 3), axis=0)) > 1
    assert _sha256(environment.source) == source_hash
    assert _sha256(environment.completion_mask) == mask_hash
    assert candidate.shape == (15, 12, 3)


@pytest.mark.parametrize("mode", ["unchanged", "constant"])
def test_hard_reject_skips_judge(
    environment: _Environment,
    mode: str,
) -> None:
    class HardFailureBackend(_Backend):
        def complete(self, **kwargs: object) -> Image.Image:
            self.calls.append(dict(kwargs))
            input_rgb = kwargs["input_rgb"]
            assert isinstance(input_rgb, Image.Image)
            if mode == "unchanged":
                return input_rgb.copy()
            return Image.new("RGB", input_rgb.size, (77, 77, 77))

    backend = HardFailureBackend()
    judge = _Judge([_accept()])
    _run(
        environment,
        backend=backend,
        judge=judge,
        seeds=(0,),
        compositing_mode="explicit_mask",
    )
    attempt = _result(environment)["attempts"][0]

    assert attempt["judge_status"] == "not_run"
    assert attempt["judge_verdict"] == "reject"
    assert judge.calls == []
    expected = (
        "candidate_unchanged_inside_completion_mask"
        if mode == "unchanged"
        else "candidate_completion_is_constant"
    )
    assert expected in attempt["hard_check"]["reasons"]


def test_judge_reject_continues_and_earliest_accept_is_selected(
    environment: _Environment,
) -> None:
    judge = _Judge([_reject(identity_preserved=False), _accept()])
    backend, judge = _run(environment, judge=judge)
    result = _result(environment)

    assert len(backend.calls) == 2 and len(judge.calls) == 2
    assert [attempt["judge_verdict"] for attempt in result["attempts"]] == [
        "reject",
        "accept",
    ]
    assert result["accepted_candidate"]["candidate_id"] == "qwen_seed_17"


def test_all_rejected_has_no_fallback(environment: _Environment) -> None:
    _run(
        environment,
        judge=_Judge([_reject(), _reject(identity_preserved=False)]),
    )
    result = _result(environment)

    assert result["status"] == "rejected"
    assert result["accepted_candidate"] is None
    assert len(result["attempts"]) == 2


def test_summary_counts_hard_and_judge_rejections_deterministically(
    environment: _Environment,
) -> None:
    class MixedBackend(_Backend):
        def complete(self, **kwargs: object) -> Image.Image:
            self.calls.append(dict(kwargs))
            input_rgb = kwargs["input_rgb"]
            seed = kwargs["seed"]
            assert isinstance(input_rgb, Image.Image) and isinstance(seed, int)
            if seed == 0:
                return input_rgb.copy()
            return _gradient(input_rgb.size, offset=seed)

    _run(
        environment,
        backend=MixedBackend(),
        judge=_Judge([_reject(identity_preserved=False, boundary_clean=False)]),
    )
    summary = json.loads(
        (environment.output_root / "benchmark_summary.json").read_text()
    )

    assert summary == {
        "backend": QWEN_COMPLETION_BACKEND,
        "processed": 1,
        "accepted": 0,
        "rejected": 1,
        "hard_check_rejection_counts": {
            "candidate_unchanged_outside_visible_region": 1,
        },
        "judge_rejection_flag_counts": {
            "boundary_clean": 1,
            "completion_useful": 1,
            "identity_preserved": 1,
        },
        "compositing_mode_counts": {"whole_canvas": 1},
    }


def test_summary_counts_explicit_mask_mode(environment: _Environment) -> None:
    _run(environment, seeds=(0,), compositing_mode="explicit_mask")
    summary = json.loads(
        (environment.output_root / "benchmark_summary.json").read_text()
    )

    assert summary["compositing_mode_counts"] == {"explicit_mask": 1}


def test_existing_root_is_rejected_without_calls(environment: _Environment) -> None:
    environment.output_root.mkdir()
    backend = _Backend()
    judge = _Judge([_accept()])

    with pytest.raises(FileExistsError, match="already exists"):
        _run(environment, backend=backend, judge=judge, seeds=(0,))

    assert backend.calls == [] and judge.calls == []


def test_localized_existing_root_is_rejected_without_calls(
    environment: _Environment,
) -> None:
    environment.output_root.mkdir()
    backend = _Backend()
    judge = _Judge([_accept()])

    with pytest.raises(FileExistsError, match="already exists"):
        _run_localized(environment, backend=backend, judge=judge)

    assert backend.calls == [] and judge.calls == []


def test_whole_canvas_preserves_source_and_context_files(
    environment: _Environment,
) -> None:
    context = environment.source.parent / "context.png"
    Image.new("RGB", (12, 10), (231, 233, 235)).save(context, format="PNG")
    payload = json.loads(environment.manifest.read_text(encoding="utf-8"))
    payload["context_rgb_path"] = str(context)
    payload["completion_mask_path"] = None
    environment.manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    source_sha256 = _sha256(environment.source)
    context_sha256 = _sha256(context)

    _run(environment, seeds=(0,))
    result = _result(environment)

    assert _sha256(environment.source) == source_sha256
    assert _sha256(context) == context_sha256
    assert result["source_sha256"] == source_sha256
    assert result["context_sha256"] == context_sha256
    assert (environment.output_root / "sample-1" / "context_rgb.png").exists()


def test_sample_publication_is_transactional_on_metadata_failure(
    environment: _Environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write_json = completion_module.write_json_atomic

    def fail_review(path: Path, payload: object) -> None:
        if path.name.startswith("review_"):
            raise OSError("simulated metadata failure")
        original_write_json(path, payload)

    monkeypatch.setattr(completion_module, "write_json_atomic", fail_review)

    with pytest.raises(OSError, match="simulated metadata failure"):
        _run(environment, seeds=(0,))

    assert not (environment.output_root / "sample-1").exists()
    assert not list(environment.output_root.glob(".sample-1.tmp-*"))


def test_localized_publication_is_transactional_on_review_failure(
    environment: _Environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write_json = qwen_module.write_json_atomic

    def fail_review(path: Path, payload: object) -> None:
        if path.name.startswith("review_"):
            raise OSError("simulated localized metadata failure")
        original_write_json(path, payload)

    monkeypatch.setattr(qwen_module, "write_json_atomic", fail_review)

    with pytest.raises(OSError, match="simulated localized metadata failure"):
        _run_localized(environment)

    assert not (environment.output_root / "sample-1").exists()
    assert not list(environment.output_root.glob(".sample-1.tmp-*"))


def test_benchmark_does_not_touch_production_roots(environment: _Environment) -> None:
    sentinels = []
    for name in ("r2v_v3_runs", "r2v_v3_datasets", "selected"):
        sentinel = environment.data_root / name / "keep.txt"
        sentinel.parent.mkdir(parents=True)
        sentinel.write_text("unchanged", encoding="utf-8")
        sentinels.append(sentinel)

    _run(environment, seeds=(0,))

    assert all(path.read_text(encoding="utf-8") == "unchanged" for path in sentinels)


def test_cli_has_qwen_defaults_and_no_powerpaint_arguments() -> None:
    parser = _parser()
    arguments = parser.parse_args(
        [
            "--manifest",
            "/tmp/manifest.jsonl",
            "--benchmark-root",
            "/tmp/benchmark",
            "--judge-model",
            "local-qwen-judge",
        ]
    )
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }

    assert arguments.model_path == DEFAULT_QWEN_MODEL_PATH
    assert arguments.model_min_side == 1024
    assert arguments.prompt is None
    assert arguments.negative_prompt is None
    assert arguments.mode == "localized_raw"
    assert arguments.seeds is None
    assert {"--powerpaint-repo", "--checkpoint-dir", "--strategy", "--fitting-degree"}.isdisjoint(
        option_strings
    )

    explicit_arguments = parser.parse_args(
        [
            "--manifest",
            "/tmp/manifest.jsonl",
            "--benchmark-root",
            "/tmp/benchmark",
            "--judge-model",
            "local-qwen-judge",
            "--mode",
            "explicit_mask",
        ]
    )
    assert explicit_arguments.mode == "explicit_mask"
    assert "--compositing-mode" not in option_strings


def test_duplicate_or_invalid_seeds_fail_before_root(environment: _Environment) -> None:
    for index, seeds in enumerate(((), (0, 0), (-1,))):
        output_root = environment.benchmark_base / f"invalid-seed-{index}"
        with pytest.raises(ValueError, match="seeds"):
            run_qwen_reference_completion_benchmark(
                manifest_path=environment.manifest,
                benchmark_root=output_root,
                config=_config(environment),
                backend=_Backend(),
                judge=_Judge([_accept()]),
                seeds=seeds,
            )
        assert not output_root.exists()
