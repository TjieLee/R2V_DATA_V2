from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import BadRequestError
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.reference_completion_benchmark as benchmark_module
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.reference_completion_benchmark import (
    CompletionManifestRecord,
    PowerPaintV21CompletionConfig,
    PowerPaintV21ReferenceCompletionBackend,
    QwenReferenceCompletionJudge,
    ReferenceCompletionJudgeFailure,
    ReferenceCompletionReview,
    build_completion_canvas,
    build_completion_negative_prompt,
    build_completion_prompt,
    build_model_space_transform,
    build_powerpaint_task_prompts,
    load_completion_manifest,
    resize_completion_inputs,
    run_reference_completion_benchmark,
    validate_powerpaint_v21_layout,
)


@dataclass(frozen=True)
class _Environment:
    data_root: Path
    benchmark_base: Path
    output_root: Path
    manifest: Path
    source: Path
    context: Path


def _source_image(
    *,
    size: tuple[int, int] = (12, 10),
    alpha_value: int = 255,
) -> Image.Image:
    width, height = size
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :, :3] = (7, 11, 13)
    pixels[2:7, 3:9, :3] = (191, 83, 37)
    pixels[2:7, 3:9, 3] = alpha_value
    return Image.fromarray(pixels)


def _write_source(
    path: Path,
    *,
    size: tuple[int, int] = (12, 10),
    alpha_value: int = 255,
    mode: str = "RGBA",
) -> None:
    image = _source_image(size=size, alpha_value=alpha_value)
    if mode != "RGBA":
        image = image.convert(mode)
    image.save(path, format="PNG")


def _write_manifest(
    path: Path,
    source: Path,
    *,
    context: Path | None = None,
    completion_sides: list[str] | None = None,
    completion_start_ratio: float = 0.5,
    sample_id: str = "sample-1",
) -> None:
    payload = {
        "sample_id": sample_id,
        "clip_uid": "clip-1",
        "entity_id": "e1",
        "reference_type": "subject",
        "entity_phrase": "a person in a red coat",
        "source_rgba_path": str(source),
        "context_rgb_path": str(context) if context is not None else None,
        "completion_sides": completion_sides or ["bottom"],
        "completion_start_ratio": completion_start_ratio,
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


@pytest.fixture
def environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Environment:
    data_root = (tmp_path / "workspace" / "data").resolve()
    benchmark_base = data_root / "reference_completion_benchmarks"
    output_root = benchmark_base / "unit-test"
    input_root = data_root / "inputs"
    input_root.mkdir(parents=True)
    benchmark_base.mkdir(parents=True)
    monkeypatch.setattr(benchmark_module, "ALLOWED_INPUT_ROOT", data_root)
    monkeypatch.setattr(
        benchmark_module,
        "ALLOWED_BENCHMARK_ROOT",
        benchmark_base,
    )
    source = input_root / "entity.png"
    context = input_root / "context.png"
    _write_source(source)
    Image.new("RGB", (12, 10), (43, 67, 89)).save(context, format="PNG")
    manifest = data_root / "completion.jsonl"
    _write_manifest(manifest, source)
    return _Environment(
        data_root=data_root,
        benchmark_base=benchmark_base,
        output_root=output_root,
        manifest=manifest,
        source=source,
        context=context,
    )


def _config(tmp_path: Path, **updates: object) -> PowerPaintV21CompletionConfig:
    values: dict[str, object] = {
        "powerpaint_repo_path": tmp_path / "PowerPaint",
        "checkpoint_dir": tmp_path / "PowerPaint-v2-1",
        "canvas_expand_ratio": 0.5,
        "lateral_padding_ratio": 0.2,
        "model_min_side": 64,
        "model_multiple": 8,
    }
    values.update(updates)
    return PowerPaintV21CompletionConfig(**values)


def _gradient(size: tuple[int, int]) -> Image.Image:
    width, height = size
    yy, xx = np.indices((height, width))
    pixels = np.stack(
        (
            (xx * 17 + yy * 3 + 13) % 256,
            (yy * 29 + xx * 5 + 31) % 256,
            ((xx + yy) * 11 + 47) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    return Image.fromarray(pixels)


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
        "reason": "The same entity is completed cleanly.",
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
    fail: BaseException | None = None
    constant: bool = False
    calls: list[dict[str, object]] = field(default_factory=list)

    def complete(self, **kwargs: object) -> Image.Image:
        self.calls.append(dict(kwargs))
        if self.fail is not None:
            raise self.fail
        input_rgb = kwargs["input_rgb"]
        assert isinstance(input_rgb, Image.Image)
        if self.constant:
            return Image.new("RGB", input_rgb.size, (90, 90, 90))
        return _gradient(input_rgb.size)


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
    strategies: tuple[str, ...] = ("text_guided", "shape_guided"),
    seeds: tuple[int, ...] = (0, 17),
) -> tuple[_Backend, _Judge]:
    selected_backend = backend or _Backend()
    selected_judge = judge or _Judge([_accept() for _ in range(4)])
    run_reference_completion_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        config=_config(environment.data_root),
        backend=selected_backend,
        judge=selected_judge,
        strategies=strategies,
        seeds=seeds,
    )
    return selected_backend, selected_judge


def test_manifest_requires_explicit_completion_sides(environment: _Environment) -> None:
    payload = json.loads(environment.manifest.read_text())
    del payload["completion_sides"]
    environment.manifest.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="completion_sides"):
        load_completion_manifest(environment.manifest)


@pytest.mark.parametrize("sides", [[], ["bottom", "bottom"]])
def test_manifest_rejects_empty_or_duplicate_sides(
    environment: _Environment,
    sides: list[str],
) -> None:
    _write_manifest(environment.manifest, environment.source, completion_sides=sides)
    if not sides:
        payload = json.loads(environment.manifest.read_text())
        payload["completion_sides"] = []
        environment.manifest.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="completion_sides"):
        load_completion_manifest(environment.manifest)


def test_manifest_sides_have_stable_canonical_order() -> None:
    record = CompletionManifestRecord.model_validate(
        {
            "sample_id": "s1",
            "clip_uid": "c1",
            "entity_id": "e1",
            "reference_type": "object",
            "entity_phrase": "a device",
            "source_rgba_path": "/tmp/source.png",
            "context_rgb_path": None,
            "completion_sides": ["right", "top", "left"],
            "completion_start_ratio": 0.5,
        }
    )

    assert record.completion_sides == ("top", "left", "right")


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf"), float("nan")])
def test_manifest_rejects_invalid_start_ratio(value: float) -> None:
    with pytest.raises(ValidationError, match="completion_start_ratio"):
        CompletionManifestRecord.model_validate(
            {
                "sample_id": "s1",
                "clip_uid": "c1",
                "entity_id": "e1",
                "reference_type": "object",
                "entity_phrase": "a device",
                "source_rgba_path": "/tmp/source.png",
                "context_rgb_path": None,
                "completion_sides": ["bottom"],
                "completion_start_ratio": value,
            }
        )


@pytest.mark.parametrize("source_kind", ["rgb_png", "jpeg_path"])
def test_source_must_be_rgba_png(
    environment: _Environment,
    source_kind: str,
) -> None:
    if source_kind == "rgb_png":
        _write_source(environment.source, mode="RGB")
    else:
        jpeg = environment.source.with_suffix(".jpg")
        Image.new("RGB", (12, 10), (1, 2, 3)).save(jpeg, format="JPEG")
        _write_manifest(environment.manifest, jpeg)

    with pytest.raises(ValueError, match=r"RGBA PNG|\.png path"):
        _run(environment)


def test_source_alpha_must_be_binary(environment: _Environment) -> None:
    _write_source(environment.source, alpha_value=128)

    with pytest.raises(ValueError, match="only 0 and 255"):
        _run(environment)


def test_source_alpha_must_be_nonempty(environment: _Environment) -> None:
    Image.fromarray(np.zeros((10, 12, 4), dtype=np.uint8)).save(
        environment.source,
        format="PNG",
    )

    with pytest.raises(ValueError, match="foreground is empty"):
        _run(environment)


@pytest.mark.parametrize("failure", ["size", "mode"])
def test_context_must_be_matching_rgb(
    environment: _Environment,
    failure: str,
) -> None:
    if failure == "size":
        image = Image.new("RGB", (13, 10), (1, 2, 3))
    else:
        image = Image.new("RGBA", (12, 10), (1, 2, 3, 255))
    image.save(environment.context, format="PNG")
    _write_manifest(
        environment.manifest,
        environment.source,
        context=environment.context,
    )

    with pytest.raises(ValueError, match="context"):
        _run(environment)


@pytest.mark.parametrize(
    ("side", "expected_size", "expected_offset"),
    [
        ("bottom", (12, 15), (0, 0)),
        ("top", (12, 15), (0, 5)),
        ("left", (18, 10), (6, 0)),
        ("right", (18, 10), (0, 0)),
    ],
)
def test_directional_canvas_expansion_and_source_offset(
    tmp_path: Path,
    side: str,
    expected_size: tuple[int, int],
    expected_offset: tuple[int, int],
) -> None:
    source = _source_image()
    canvas = build_completion_canvas(
        source_rgba=source,
        context_rgb=None,
        completion_sides=(side,),
        completion_start_ratio=0.5,
        config=_config(tmp_path),
    )

    assert canvas.canvas_size == expected_size
    assert canvas.source_offset_xy == expected_offset
    assert canvas.source_size == source.size
    x, y = expected_offset
    visible = np.asarray(source.getchannel("A")) == 255
    output = np.asarray(canvas.baseline_rgb)[y : y + 10, x : x + 12]
    assert np.array_equal(output[visible], np.asarray(source)[:, :, :3][visible])


def test_multiple_directions_are_unioned_without_unrequested_expansion(
    tmp_path: Path,
) -> None:
    canvas = build_completion_canvas(
        source_rgba=_source_image(),
        context_rgb=None,
        completion_sides=("top", "right"),
        completion_start_ratio=0.5,
        config=_config(tmp_path),
    )
    mask = np.asarray(canvas.completion_mask) == 255

    assert canvas.canvas_size == (18, 15)
    assert canvas.source_offset_xy == (0, 5)
    assert mask[:5].any()
    assert mask[:, 12:].any()


def test_context_only_fills_source_rectangle_background(tmp_path: Path) -> None:
    source = _source_image()
    context = Image.new("RGB", source.size, (44, 55, 66))
    canvas = build_completion_canvas(
        source_rgba=source,
        context_rgb=context,
        completion_sides=("bottom",),
        completion_start_ratio=0.5,
        config=_config(tmp_path),
    )
    baseline = np.asarray(canvas.baseline_rgb)
    alpha = np.asarray(source.getchannel("A")) == 255

    assert np.all(baseline[:10][~alpha] == (44, 55, 66))
    assert np.array_equal(baseline[:10][alpha], np.asarray(source)[:, :, :3][alpha])
    assert np.all(baseline[10:] == 255)


def test_completion_mask_is_binary_bounded_adjacent_and_excludes_visible(
    tmp_path: Path,
) -> None:
    canvas = build_completion_canvas(
        source_rgba=_source_image(),
        context_rgb=None,
        completion_sides=("bottom",),
        completion_start_ratio=0.5,
        config=_config(tmp_path),
    )
    mask = np.asarray(canvas.completion_mask)
    visible = np.asarray(canvas.visible_mask) == 255

    assert canvas.completion_mask.mode == "L"
    assert set(np.unique(mask)) == {0, 255}
    assert not np.any((mask == 255) & visible)
    assert (mask == 255).any()
    assert not (mask == 255).all()
    assert np.logical_and(
        mask == 255,
        benchmark_module._dilate_one_pixel(visible) & ~visible,
    ).any()


def test_completion_mask_cannot_equal_all_transparent_pixels(tmp_path: Path) -> None:
    source = Image.new("RGBA", (4, 4), (20, 30, 40, 255))
    with pytest.raises(ValueError, match="all transparent"):
        build_completion_canvas(
            source_rgba=source,
            context_rgb=None,
            completion_sides=("top", "bottom", "left", "right"),
            completion_start_ratio=0.0,
            config=_config(
                tmp_path,
                canvas_expand_ratio=0.5,
                lateral_padding_ratio=1.0,
            ),
        )


@pytest.mark.parametrize("canvas_size", [(17, 29), (41, 19), (31, 31)])
def test_model_resize_preserves_aspect_and_multiple_of_eight(
    canvas_size: tuple[int, int],
) -> None:
    transform = build_model_space_transform(
        canvas_size,
        model_min_side=640,
        model_multiple=8,
    )
    source_ratio = canvas_size[0] / canvas_size[1]
    model_ratio = transform.model_size[0] / transform.model_size[1]

    assert transform.model_size[0] % 8 == 0
    assert transform.model_size[1] % 8 == 0
    assert min(transform.model_size) == 640
    assert model_ratio == pytest.approx(source_ratio, rel=0.015)


def test_model_mask_resize_remains_binary(tmp_path: Path) -> None:
    canvas = build_completion_canvas(
        source_rgba=_source_image(),
        context_rgb=None,
        completion_sides=("right",),
        completion_start_ratio=0.5,
        config=_config(tmp_path),
    )
    transform = build_model_space_transform(
        canvas.canvas_size,
        model_min_side=64,
        model_multiple=8,
    )
    model_rgb, model_mask = resize_completion_inputs(canvas, transform)

    assert model_rgb.size == transform.model_size
    assert model_mask.size == transform.model_size
    assert set(np.unique(np.asarray(model_mask))).issubset({0, 255})


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (
            "text_guided",
            ("P_obj", "P_obj", "P_obj", "P_obj"),
        ),
        (
            "shape_guided",
            ("P_shape", "P_ctxt", "P_shape", "P_ctxt"),
        ),
    ],
)
def test_powerpaint_task_tokens_match_v21_strategy(
    strategy: str,
    expected: tuple[str, str, str, str],
) -> None:
    tasks = build_powerpaint_task_prompts(
        strategy=strategy,
        prompt="user prompt",
        negative_prompt="user negative",
    )

    assert (
        tasks.promptA,
        tasks.promptB,
        tasks.negative_promptA,
        tasks.negative_promptB,
    ) == expected
    assert tasks.promptU == "user prompt"
    assert tasks.negative_promptU == "user negative"


@pytest.mark.parametrize("reference_type", ["subject", "object", "group"])
def test_completion_prompts_are_nonempty_and_include_entity_phrase(
    reference_type: str,
) -> None:
    prompt = build_completion_prompt(
        entity_phrase="the original entity",
        reference_type=reference_type,
    )
    negative = build_completion_negative_prompt(reference_type)

    assert "the original entity" in prompt
    assert "completion mask" in prompt
    assert "duplicate" in negative
    assert "new foreground object" in negative


class _Generator:
    def __init__(self, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


class _Pipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        image = kwargs["image"]
        assert isinstance(image, Image.Image)
        return SimpleNamespace(images=[_gradient(image.size)])


@pytest.mark.parametrize("strategy", ["text_guided", "shape_guided"])
def test_injected_powerpaint_backend_uses_official_call_contract(
    tmp_path: Path,
    strategy: str,
) -> None:
    pipeline = _Pipeline()
    torch_module = SimpleNamespace(Generator=_Generator)
    backend = PowerPaintV21ReferenceCompletionBackend(
        _config(tmp_path, fitting_degree=0.63),
        pipeline=pipeline,
        torch_module=torch_module,
    )
    input_rgb = _gradient((16, 24))
    mask_array = np.zeros((24, 16), dtype=np.uint8)
    mask_array[8:20, 5:12] = 255

    output = backend.complete(
        input_rgb=input_rgb,
        completion_mask=Image.fromarray(mask_array),
        entity_phrase="the same person",
        reference_type="subject",
        strategy=strategy,
        seed=17,
        fitting_degree=0.63,
        prompt="complete the person",
        negative_prompt="duplicate person",
    )
    call = pipeline.calls[0]
    masked_input = np.asarray(call["image"])

    assert output.mode == "RGB"
    assert output.size == input_rgb.size
    assert call["promptU"] == "complete the person"
    assert call["negative_promptU"] == "duplicate person"
    assert call["tradoff"] == call["tradoff_nag"] == 0.63
    assert call["mask"].mode == "RGB"
    assert np.all(masked_input[mask_array == 255] == 0)
    assert np.array_equal(
        masked_input[mask_array == 0],
        np.asarray(input_rgb)[mask_array == 0],
    )
    assert call["generator"].seed == 17


@pytest.mark.parametrize(
    ("field_name", "value", "error"),
    [
        ("local_files_only", False, "local_files_only"),
        ("fitting_degree", 1.1, "fitting_degree"),
        ("canvas_expand_ratio", 0.0, "canvas_expand_ratio"),
        ("model_min_side", 0, "model_min_side"),
        ("model_multiple", 0, "model_multiple"),
        ("mask_overlap_pixels", -1, "mask_overlap_pixels"),
    ],
)
def test_powerpaint_config_validation_is_strict(
    tmp_path: Path,
    field_name: str,
    value: object,
    error: str,
) -> None:
    config = replace(_config(tmp_path), **{field_name: value})

    with pytest.raises((TypeError, ValueError), match=error):
        config.validate()


def _create_powerpaint_layout(tmp_path: Path) -> PowerPaintV21CompletionConfig:
    config = _config(tmp_path)
    repo = config.powerpaint_repo_path
    checkpoint = config.checkpoint_dir
    for relative in (
        "powerpaint/models/BrushNet_CA.py",
        "powerpaint/models/unet_2d_condition.py",
        "powerpaint/pipelines/pipeline_PowerPaint_Brushnet_CA.py",
        "powerpaint/utils/utils.py",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test fixture\n")
    (checkpoint / "realisticVisionV60B1_v51VAE").mkdir(parents=True)
    brushnet = checkpoint / "PowerPaint_Brushnet"
    brushnet.mkdir(parents=True)
    (brushnet / "diffusion_pytorch_model.safetensors").write_bytes(b"weights")
    (brushnet / "pytorch_model.bin").write_bytes(b"weights")
    return config


def test_powerpaint_v21_layout_accepts_complete_local_tree(tmp_path: Path) -> None:
    paths = validate_powerpaint_v21_layout(_create_powerpaint_layout(tmp_path))

    assert paths["base_model"].is_dir()
    assert paths["brushnet_weights"].is_file()
    assert paths["pipeline_source"].is_file()


@pytest.mark.parametrize(
    "missing_key",
    ["base_model", "brushnet_weights", "text_encoder_weights", "pipeline_source"],
)
def test_powerpaint_v21_layout_missing_artifact_fails_closed(
    tmp_path: Path,
    missing_key: str,
) -> None:
    config = _create_powerpaint_layout(tmp_path)
    paths = validate_powerpaint_v21_layout(config)
    path = paths[missing_key]
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()

    with pytest.raises(FileNotFoundError, match="PowerPaint v2-1"):
        validate_powerpaint_v21_layout(config)


def test_powerpaint_loader_uses_custom_local_only_v21_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _create_powerpaint_layout(tmp_path)
    records: dict[str, object] = {"pretrained": [], "offload": False}

    class _LoadedComponent:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: object) -> _LoadedComponent:
            records["pretrained"].append((cls.__name__, path, dict(kwargs)))
            return cls()

        def load_state_dict(self, state: object) -> None:
            records["text_state"] = state

    class _BrushNet:
        @classmethod
        def from_unet(cls, unet: object) -> _BrushNet:
            records["brushnet_unet"] = unet
            return cls()

    class _TokenizerWrapper:
        def __init__(self, tokenizer: object) -> None:
            self.tokenizer = tokenizer

    def _add_tokens(
        tokenizer: object,
        text_encoder: object,
        **kwargs: object,
    ) -> None:
        records["tokens"] = (tokenizer, text_encoder, kwargs)

    class _LoadedPipeline:
        def __init__(self) -> None:
            self.scheduler = SimpleNamespace(config={"name": "scheduler"})

        @classmethod
        def from_pretrained(
            cls,
            path: str,
            **kwargs: object,
        ) -> _LoadedPipeline:
            records["pipeline"] = (path, dict(kwargs))
            return cls()

        def set_progress_bar_config(self, *, disable: bool) -> None:
            records["progress_disabled"] = disable

        def enable_model_cpu_offload(self) -> None:
            records["offload"] = True

    class _Scheduler:
        @classmethod
        def from_config(cls, config_value: object) -> object:
            records["scheduler_config"] = config_value
            return SimpleNamespace(config=config_value)

    def _torch_load(
        path: str,
        *,
        map_location: str,
        weights_only: bool,
    ) -> dict[str, object]:
        records["torch_load"] = (path, map_location, weights_only)
        return {"state_dict": {"embedding": "local"}}

    torch_module = types.ModuleType("torch")
    torch_module.float16 = "float16"
    torch_module.load = _torch_load
    torch_module.cuda = SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    diffusers_module = types.ModuleType("diffusers")
    diffusers_module.UniPCMultistepScheduler = _Scheduler
    transformers_module = types.ModuleType("transformers")
    transformers_module.CLIPTextModel = type(
        "CLIPTextModel",
        (_LoadedComponent,),
        {},
    )
    transformers_module.CLIPTokenizer = type(
        "CLIPTokenizer",
        (_LoadedComponent,),
        {},
    )
    safetensors_module = types.ModuleType("safetensors")
    safetensors_torch_module = types.ModuleType("safetensors.torch")

    def _load_model(model: object, path: str) -> None:
        records["brushnet_weights"] = (model, path)

    safetensors_torch_module.load_model = _load_model
    for name, module in (
        ("torch", torch_module),
        ("diffusers", diffusers_module),
        ("transformers", transformers_module),
        ("safetensors", safetensors_module),
        ("safetensors.torch", safetensors_torch_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    custom_modules = {
        "powerpaint.models.BrushNet_CA": SimpleNamespace(BrushNetModel=_BrushNet),
        "powerpaint.models.unet_2d_condition": SimpleNamespace(
            UNet2DConditionModel=type(
                "UNet2DConditionModel",
                (_LoadedComponent,),
                {},
            )
        ),
        "powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA": SimpleNamespace(
            StableDiffusionPowerPaintBrushNetPipeline=_LoadedPipeline
        ),
        "powerpaint.utils.utils": SimpleNamespace(
            TokenizerWrapper=_TokenizerWrapper,
            add_tokens=_add_tokens,
        ),
    }
    monkeypatch.setattr(
        benchmark_module.importlib,
        "import_module",
        lambda name: custom_modules[name],
    )

    backend = PowerPaintV21ReferenceCompletionBackend(config)
    backend._ensure_loaded()

    assert records["offload"] is True
    assert records["progress_disabled"] is True
    assert all(
        kwargs["local_files_only"] is True for _, _, kwargs in records["pretrained"]
    )
    pipeline_kwargs = records["pipeline"][1]
    assert pipeline_kwargs["local_files_only"] is True
    assert pipeline_kwargs["safety_checker"] is None
    assert pipeline_kwargs["feature_extractor"] is None
    token_kwargs = records["tokens"][2]
    assert token_kwargs["placeholder_tokens"] == ["P_ctxt", "P_shape", "P_obj"]
    assert token_kwargs["num_vectors_per_token"] == 10
    assert records["torch_load"][1:] == ("cpu", True)


def test_candidate_order_is_deterministic(environment: _Environment) -> None:
    backend, _ = _run(environment)

    assert [(call["strategy"], call["seed"]) for call in backend.calls] == [
        ("text_guided", 0),
        ("text_guided", 17),
        ("shape_guided", 0),
        ("shape_guided", 17),
    ]


def test_first_accepted_candidate_wins_but_all_candidates_are_retained(
    environment: _Environment,
) -> None:
    judge = _Judge([_reject(), _accept(), _accept(), _accept()])

    _run(environment, judge=judge)

    result = json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )
    assert result["status"] == "accepted"
    assert result["accepted_candidate"]["strategy"] == "text_guided"
    assert result["accepted_candidate"]["seed"] == 17
    assert len(result["attempts"]) == 4
    assert len(judge.calls) == 4


def test_all_candidates_rejected_without_fallback(
    environment: _Environment,
) -> None:
    judge = _Judge([_reject() for _ in range(4)])
    stats = run_reference_completion_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        config=_config(environment.data_root),
        backend=_Backend(),
        judge=judge,
    )

    result = json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )
    assert stats.to_dict() == {"processed": 1, "accepted": 0, "rejected": 1}
    assert result["status"] == "rejected"
    assert result["accepted_candidate"] is None


def test_final_candidate_restores_visible_and_outside_mask_pixels(
    environment: _Environment,
) -> None:
    _run(
        environment,
        judge=_Judge([_accept()]),
        strategies=("text_guided",),
        seeds=(0,),
    )
    sample = environment.output_root / "sample-1"
    with Image.open(sample / "candidate_text_guided_seed_0.png") as opened:
        candidate = np.asarray(opened.convert("RGB"))
    with Image.open(sample / "baseline_canvas.png") as opened:
        baseline = np.asarray(opened.convert("RGB"))
    with Image.open(sample / "visible_mask.png") as opened:
        visible = np.asarray(opened.convert("L")) == 255
    with Image.open(sample / "completion_mask.png") as opened:
        completion = np.asarray(opened.convert("L")) == 255

    assert np.array_equal(candidate[visible], baseline[visible])
    assert np.array_equal(candidate[~completion], baseline[~completion])


def test_candidate_restores_completion_canvas_size(environment: _Environment) -> None:
    _run(
        environment,
        judge=_Judge([_accept()]),
        strategies=("text_guided",),
        seeds=(0,),
    )
    sample = environment.output_root / "sample-1"
    with Image.open(sample / "candidate_text_guided_seed_0.png") as candidate:
        assert candidate.size == (12, 15)
        assert candidate.mode == "RGB"


def test_unchanged_candidate_fails_hard_check_without_qwen(
    environment: _Environment,
) -> None:
    source = benchmark_module._load_source_rgba(environment.source)[0]
    config = _config(environment.data_root)
    canvas = build_completion_canvas(
        source_rgba=source,
        context_rgb=None,
        completion_sides=("bottom",),
        completion_start_ratio=0.5,
        config=config,
    )
    transform = build_model_space_transform(
        canvas.canvas_size,
        model_min_side=config.model_min_side,
        model_multiple=config.model_multiple,
    )
    unchanged_model = canvas.baseline_rgb.resize(
        transform.model_size,
        resample=Image.Resampling.LANCZOS,
    )
    judge = _Judge([_accept()])
    backend = _Backend()
    backend.complete = lambda **_: unchanged_model.copy()  # type: ignore[method-assign]

    _run(
        environment,
        backend=backend,
        judge=judge,
        strategies=("text_guided",),
        seeds=(0,),
    )

    result = json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )
    attempt = result["attempts"][0]
    assert attempt["hard_check"]["status"] == "failed"
    assert (
        "candidate_unchanged_inside_completion_mask" in attempt["hard_check"]["reasons"]
    )
    assert judge.calls == []


def test_constant_completion_fails_hard_check_without_qwen(
    environment: _Environment,
) -> None:
    judge = _Judge([_accept()])

    _run(
        environment,
        backend=_Backend(constant=True),
        judge=judge,
        strategies=("text_guided",),
        seeds=(0,),
    )

    result = json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )
    assert (
        "candidate_completion_is_constant"
        in result["attempts"][0]["hard_check"]["reasons"]
    )
    assert judge.calls == []


@pytest.mark.parametrize(
    "failed_flag",
    ["exactly_one_entity", "identity_preserved", "completion_useful"],
)
def test_judge_quality_flag_failure_requires_reject(failed_flag: str) -> None:
    payload = _review_payload(
        verdict="reject",
        **{failed_flag: False},
    )
    review = ReferenceCompletionReview.model_validate(payload)

    assert review.verdict == "reject"


def test_review_schema_is_strict_and_verdict_is_deterministic() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ReferenceCompletionReview.model_validate(_review_payload(extra="bad"))
    with pytest.raises(ValidationError, match="if and only if"):
        ReferenceCompletionReview.model_validate(_review_payload(verdict="reject"))


class _Completions:
    def __init__(
        self,
        responses: list[dict[str, object] | str],
        *,
        strict_failure: bool = False,
    ) -> None:
        self.responses = iter(responses)
        self.strict_failure = strict_failure
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(dict(kwargs))
        if self.strict_failure and len(self.calls) == 1:
            raise BadRequestError(
                "json_schema unsupported",
                response=httpx.Response(
                    400,
                    request=httpx.Request(
                        "POST",
                        "http://127.0.0.1:8000/v1/chat/completions",
                    ),
                ),
                body={},
            )
        response = next(self.responses)
        raw = response if isinstance(response, str) else json.dumps(response)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw))]
        )


def _qwen_judge(
    completions: _Completions,
    *,
    repair_retries: int = 1,
) -> QwenReferenceCompletionJudge:
    return QwenReferenceCompletionJudge(
        QwenServiceConfig(model="local-qwen-vl", max_tokens=500),
        repair_retries=repair_retries,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )


def _review_with_qwen(
    judge: QwenReferenceCompletionJudge,
) -> ReferenceCompletionReview:
    source = _source_image()
    canvas = Image.new("RGB", (12, 15), (255, 255, 255))
    mask_array = np.zeros((15, 12), dtype=np.uint8)
    mask_array[6:, 2:10] = 255
    return judge.review(
        source_rgba=source,
        completion_canvas=canvas,
        completion_mask=Image.fromarray(mask_array),
        candidate_rgb=_gradient(canvas.size),
        entity_phrase="a person in a red coat",
        reference_type="subject",
    )


def test_qwen_judge_requests_strict_schema_with_four_images() -> None:
    completions = _Completions([_review_payload()])

    review = _review_with_qwen(_qwen_judge(completions))

    assert review.verdict == "accept"
    response_format = completions.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    content = completions.calls[0]["messages"][1]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 4


def test_qwen_judge_falls_back_to_json_object() -> None:
    completions = _Completions([_review_payload()], strict_failure=True)

    review = _review_with_qwen(_qwen_judge(completions, repair_retries=0))

    assert review.verdict == "accept"
    assert [call["response_format"]["type"] for call in completions.calls] == [
        "json_schema",
        "json_object",
    ]


def test_qwen_judge_repairs_malformed_output() -> None:
    completions = _Completions(["not json", _review_payload()])

    review = _review_with_qwen(_qwen_judge(completions))

    assert review.verdict == "accept"
    repair_text = completions.calls[1]["messages"][1]["content"][0]["text"]
    assert "Original invalid response" in repair_text
    assert "not json" in repair_text


def test_malformed_qwen_review_fails_closed() -> None:
    completions = _Completions(["not json", "still not json"])

    with pytest.raises(ReferenceCompletionJudgeFailure):
        _review_with_qwen(_qwen_judge(completions))


def test_runner_fails_closed_when_judge_raises(
    environment: _Environment,
) -> None:
    stats = run_reference_completion_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        config=_config(environment.data_root),
        backend=_Backend(),
        judge=_Judge([ReferenceCompletionJudgeFailure(raw_responses=[], issues=[])]),
        strategies=("text_guided",),
        seeds=(0,),
    )

    result = json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )
    assert stats.rejected == 1
    assert result["attempts"][0]["judge_status"] == "failed_closed"
    assert result["accepted_candidate"] is None


def test_source_and_context_are_copied_without_modification(
    environment: _Environment,
) -> None:
    _write_manifest(
        environment.manifest,
        environment.source,
        context=environment.context,
    )
    source_before = environment.source.read_bytes()
    context_before = environment.context.read_bytes()

    _run(
        environment,
        judge=_Judge([_accept()]),
        strategies=("text_guided",),
        seeds=(0,),
    )

    sample = environment.output_root / "sample-1"
    result = json.loads((sample / "result.json").read_text())
    assert environment.source.read_bytes() == source_before
    assert environment.context.read_bytes() == context_before
    assert (sample / "source_rgba.png").read_bytes() == source_before
    assert result["source_sha256"] == hashlib.sha256(source_before).hexdigest()
    assert result["context_sha256"] == hashlib.sha256(context_before).hexdigest()


def test_sample_publication_is_transactional(environment: _Environment) -> None:
    backend = _Backend(fail=AssertionError("stop before publication"))

    with pytest.raises(AssertionError, match="stop before publication"):
        _run(
            environment,
            backend=backend,
            judge=_Judge([_accept()]),
            strategies=("text_guided",),
            seeds=(0,),
        )

    assert environment.output_root.exists()
    assert not (environment.output_root / "sample-1").exists()
    assert list(environment.output_root.glob(".sample-1.tmp-*")) == []


def test_existing_benchmark_root_fails_without_overwrite(
    environment: _Environment,
) -> None:
    environment.output_root.mkdir()
    sentinel = environment.output_root / "sentinel.txt"
    sentinel.write_text("keep")

    with pytest.raises(FileExistsError, match="already exists"):
        _run(environment)

    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize(
    "relative",
    [
        "r2v_v3_runs/run-1",
        "r2v_v3_datasets/dataset-1",
        "selected/run-1",
    ],
)
def test_production_output_paths_are_rejected(
    environment: _Environment,
    relative: str,
) -> None:
    output = environment.benchmark_base / relative

    with pytest.raises(ValueError, match="forbidden production"):
        run_reference_completion_benchmark(
            manifest_path=environment.manifest,
            benchmark_root=output,
            config=_config(environment.data_root),
            backend=_Backend(),
            judge=_Judge([_accept()]),
            strategies=("text_guided",),
            seeds=(0,),
        )


def test_source_path_escape_is_rejected_before_output_creation(
    environment: _Environment,
    tmp_path: Path,
) -> None:
    escaped = tmp_path / "escaped.png"
    _write_source(escaped)
    _write_manifest(environment.manifest, escaped)

    with pytest.raises(ValueError, match="must remain under"):
        _run(environment)

    assert not environment.output_root.exists()


def test_duplicate_sample_ids_are_rejected_before_output_creation(
    environment: _Environment,
) -> None:
    line = environment.manifest.read_text()
    environment.manifest.write_text(line + line)

    with pytest.raises(ValueError, match="duplicate completion sample_id"):
        _run(environment)

    assert not environment.output_root.exists()


def test_invalid_strategy_or_seed_fails_before_output_creation(
    environment: _Environment,
) -> None:
    with pytest.raises(ValueError, match="unsupported completion strategy"):
        _run(environment, strategies=("unknown",), seeds=(0,))
    assert not environment.output_root.exists()

    with pytest.raises(ValueError, match="non-negative"):
        _run(environment, strategies=("text_guided",), seeds=(-1,))
    assert not environment.output_root.exists()


def test_benchmark_writes_only_isolated_sample_artifacts(
    environment: _Environment,
) -> None:
    sentinels: list[Path] = []
    for relative in (
        "r2v_v3_runs/run-1/clips/clip-1/clip.json",
        "r2v_v3_datasets/dataset-1/dataset.json",
        "selected/reference.png",
    ):
        path = environment.data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("sentinel")
        sentinels.append(path)

    _run(
        environment,
        judge=_Judge([_accept()]),
        strategies=("text_guided",),
        seeds=(0,),
    )

    assert all(path.read_text() == "sentinel" for path in sentinels)
    sample_files = {
        path.name for path in (environment.output_root / "sample-1").iterdir()
    }
    assert sample_files == {
        "source_rgba.png",
        "source_white.png",
        "baseline_canvas.png",
        "visible_mask.png",
        "completion_mask.png",
        "candidate_text_guided_seed_0.png",
        "review_text_guided_seed_0.json",
        "result.json",
    }
