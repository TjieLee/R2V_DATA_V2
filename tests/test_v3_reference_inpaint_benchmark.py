from __future__ import annotations

import hashlib
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from openai import BadRequestError
from PIL import Image
from pydantic import ValidationError

import r2v_data_v2.v3.reference_inpaint_benchmark as benchmark_module
from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.reference_inpaint_benchmark import (
    QwenImageEditBenchmarkConfig,
    QwenImageEditReferenceBackgroundInpainter,
    QwenReferenceBackgroundJudge,
    ReferenceBackgroundJudgeFailure,
    ReferenceBackgroundReview,
    build_reference_background_prompt,
    run_reference_inpaint_benchmark,
)
from run_pipeline_v3 import STAGE_ORDER


@dataclass(frozen=True)
class _Environment:
    data_root: Path
    benchmark_base: Path
    output_root: Path
    manifest: Path
    source: Path


@pytest.fixture
def environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _Environment:
    data_root = (tmp_path / "workspace" / "data").resolve()
    benchmark_base = data_root / "reference_inpaint_benchmarks"
    output_root = benchmark_base / "unit-test"
    input_root = data_root / "inputs"
    input_root.mkdir(parents=True)
    benchmark_base.mkdir(parents=True)
    monkeypatch.setattr(
        benchmark_module,
        "ALLOWED_INPUT_ROOT",
        data_root,
    )
    monkeypatch.setattr(
        benchmark_module,
        "ALLOWED_BENCHMARK_ROOT",
        benchmark_base,
    )
    source = input_root / "entity.png"
    _write_source(source)
    manifest = data_root / "benchmark.jsonl"
    _write_manifest(manifest, source)
    return _Environment(
        data_root=data_root,
        benchmark_base=benchmark_base,
        output_root=output_root,
        manifest=manifest,
        source=source,
    )


def _write_source(
    path: Path,
    *,
    size: tuple[int, int] = (11, 7),
    alpha_value: int = 255,
    mode: str = "RGBA",
) -> None:
    width, height = size
    if mode == "RGBA":
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        pixels[:, :, :3] = (23, 41, 67)
        pixels[2:6, 3:8, :3] = (190, 80, 35)
        pixels[2:6, 3:8, 3] = alpha_value
    else:
        pixels = np.full((height, width, 3), (23, 41, 67), dtype=np.uint8)
    Image.fromarray(pixels).save(path, format="PNG")


def _write_manifest(
    path: Path,
    source: Path,
    *,
    sample_id: str = "sample-1",
) -> None:
    payload = {
        "sample_id": sample_id,
        "clip_uid": "clip-1",
        "entity_id": "e1",
        "reference_type": "subject",
        "entity_phrase": "a person in a red coat",
        "source_rgba_path": str(source),
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _gradient(size: tuple[int, int]) -> Image.Image:
    width, height = size
    yy, xx = np.indices((height, width))
    pixels = np.stack(
        (
            (xx * 17 + 13) % 256,
            (yy * 29 + 31) % 256,
            ((xx + yy) * 11 + 47) % 256,
        ),
        axis=2,
    ).astype(np.uint8)
    return Image.fromarray(pixels)


def _accept(
    reason: str = "The candidate is clean and usable.",
) -> ReferenceBackgroundReview:
    return ReferenceBackgroundReview(
        verdict="accept",
        entity_unchanged=True,
        no_duplicate_entity=True,
        no_new_salient_entity=True,
        background_natural=True,
        boundary_clean=True,
        reference_usable=True,
        reason=reason,
    )


def _reject(
    reason: str = "The background is not natural.",
) -> ReferenceBackgroundReview:
    return ReferenceBackgroundReview(
        verdict="reject",
        entity_unchanged=True,
        no_duplicate_entity=True,
        no_new_salient_entity=True,
        background_natural=False,
        boundary_clean=True,
        reference_usable=True,
        reason=reason,
    )


@dataclass
class _Backend:
    outputs: dict[int, Image.Image | Exception]
    calls: list[dict[str, object]] = field(default_factory=list)

    def generate(
        self,
        *,
        source_rgba: Image.Image,
        entity_phrase: str,
        reference_type: str,
        seed: int,
        prompt: str,
    ) -> Image.Image:
        self.calls.append(
            {
                "size": source_rgba.size,
                "mode": source_rgba.mode,
                "entity_phrase": entity_phrase,
                "reference_type": reference_type,
                "seed": seed,
                "prompt": prompt,
            }
        )
        result = self.outputs[seed]
        if isinstance(result, Exception):
            raise result
        return result.copy()


@dataclass
class _Judge:
    outcomes: list[ReferenceBackgroundReview | Exception]
    calls: list[dict[str, object]] = field(default_factory=list)

    def review(
        self,
        *,
        source_rgba: Image.Image,
        candidate_rgb: Image.Image,
        alpha_mask: Image.Image,
        entity_phrase: str,
        reference_type: str,
    ) -> ReferenceBackgroundReview:
        self.calls.append(
            {
                "source_mode": source_rgba.mode,
                "candidate_mode": candidate_rgb.mode,
                "alpha_mode": alpha_mask.mode,
                "size": candidate_rgb.size,
                "entity_phrase": entity_phrase,
                "reference_type": reference_type,
            }
        )
        result = self.outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _run(
    environment: _Environment,
    *,
    backend: _Backend | QwenImageEditReferenceBackgroundInpainter | None = None,
    judge: _Judge | None = None,
) -> tuple[_Backend | QwenImageEditReferenceBackgroundInpainter, _Judge]:
    selected_backend = backend or _Backend(
        {0: _gradient((11, 7)), 17: _gradient((11, 7))}
    )
    selected_judge = judge or _Judge([_accept(), _accept()])
    run_reference_inpaint_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        backend=selected_backend,
        judge=selected_judge,
    )
    return selected_backend, selected_judge


@pytest.mark.parametrize("source_kind", ["rgb_png", "jpeg_path"])
def test_source_must_be_rgba_png(
    environment: _Environment,
    source_kind: str,
) -> None:
    if source_kind == "rgb_png":
        _write_source(environment.source, mode="RGB")
    else:
        jpeg = environment.source.with_suffix(".jpg")
        Image.new("RGB", (11, 7), (1, 2, 3)).save(jpeg, format="JPEG")
        _write_manifest(environment.manifest, jpeg)

    with pytest.raises(ValueError, match=r"RGBA PNG|\.png path"):
        _run(environment)


def test_source_alpha_must_be_binary(environment: _Environment) -> None:
    _write_source(environment.source, alpha_value=128)

    with pytest.raises(ValueError, match="only 0 and 255"):
        _run(environment)


def test_source_foreground_must_be_nonempty(environment: _Environment) -> None:
    source = np.zeros((7, 11, 4), dtype=np.uint8)
    Image.fromarray(source).save(environment.source, format="PNG")

    with pytest.raises(ValueError, match="foreground is empty"):
        _run(environment)


def test_output_size_and_rgb_png_are_preserved(environment: _Environment) -> None:
    _run(environment)

    candidate = environment.output_root / "sample-1" / "candidate_seed_0.png"
    with Image.open(candidate) as image:
        image.load()
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (11, 7)


def test_source_foreground_pixels_are_exact(environment: _Environment) -> None:
    _run(environment)

    with Image.open(environment.source) as source_image:
        source = np.asarray(source_image.convert("RGBA"))
    with Image.open(
        environment.output_root / "sample-1" / "candidate_seed_0.png"
    ) as candidate_image:
        candidate = np.asarray(candidate_image.convert("RGB"))
    foreground = source[:, :, 3] == 255
    assert np.array_equal(candidate[foreground], source[:, :, :3][foreground])


def test_model_foreground_edits_are_overwritten_by_source(
    environment: _Environment,
) -> None:
    generated = _gradient((11, 7))
    pixels = np.asarray(generated).copy()
    pixels[2:6, 3:8] = (0, 255, 0)
    backend = _Backend({0: Image.fromarray(pixels)})
    run_reference_inpaint_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        backend=backend,
        judge=_Judge([_accept()]),
        seeds=(0,),
    )

    with Image.open(environment.source) as source_image:
        source = np.asarray(source_image)
    with Image.open(
        environment.output_root / "sample-1" / "candidate_seed_0.png"
    ) as candidate_image:
        candidate = np.asarray(candidate_image)
    foreground = source[:, :, 3] == 255
    assert np.array_equal(candidate[foreground], source[:, :, :3][foreground])
    assert not np.any(np.all(candidate[foreground] == (0, 255, 0), axis=1))


def test_benchmark_passes_original_dimensions_without_resize(
    environment: _Environment,
) -> None:
    _write_source(environment.source, size=(19, 13))
    backend = _Backend({0: _gradient((19, 13))})
    run_reference_inpaint_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        backend=backend,
        judge=_Judge([_accept()]),
        seeds=(0,),
    )

    assert backend.calls[0]["size"] == (19, 13)


class _Generator:
    def __init__(self, *, device: str) -> None:
        self.device = device
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> _Generator:
        self.seed = seed
        return self


class _InjectedPipeline:
    vae_scale_factor = 8

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        self.calls.append(
            {
                "image_size": image[0].size,
                "height": height,
                "width": width,
                "prompt": prompt,
                "seed": generator.seed,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "num_images_per_prompt": num_images_per_prompt,
            }
        )
        return SimpleNamespace(images=[_gradient(image[0].size)])


def test_qwen_backend_pads_and_crops_without_resize(tmp_path: Path) -> None:
    source_path = tmp_path / "entity.png"
    _write_source(source_path, size=(19, 13))
    with Image.open(source_path) as opened:
        source = opened.convert("RGBA")
    pipeline = _InjectedPipeline()
    torch_module = SimpleNamespace(
        Generator=_Generator,
        cuda=SimpleNamespace(
            is_available=lambda: False,
            empty_cache=lambda: None,
        ),
    )
    backend = QwenImageEditReferenceBackgroundInpainter(
        QwenImageEditBenchmarkConfig(model_path=tmp_path / "unused"),
        pipeline=pipeline,
        torch_module=torch_module,
    )

    result = backend.generate(
        source_rgba=source,
        entity_phrase="a red object",
        reference_type="object",
        seed=17,
        prompt="generate background",
    )

    assert pipeline.calls[0]["image_size"] == (32, 16)
    assert pipeline.calls[0]["width"] == 32
    assert pipeline.calls[0]["height"] == 16
    assert result.size == (19, 13)
    assert result.mode == "RGB"


def test_qwen_backend_loads_local_files_only_without_lora(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []
    model_path = tmp_path / "model"
    model_path.mkdir()
    torch = types.ModuleType("torch")
    torch.bfloat16 = object()
    torch.Generator = _Generator
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )

    class FakePipeline(_InjectedPipeline):
        @classmethod
        def from_pretrained(
            cls,
            path: str,
            *,
            torch_dtype: object,
            local_files_only: bool,
        ) -> FakePipeline:
            events.append(("load", Path(path), torch_dtype, local_files_only))
            return cls()

        def set_progress_bar_config(self, *, disable: bool) -> None:
            events.append(("progress", disable))

        def to(self, device: str) -> FakePipeline:
            events.append(("device", device))
            return self

    diffusers = types.ModuleType("diffusers")
    diffusers.QwenImageEditPlusPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    source_path = tmp_path / "entity.png"
    _write_source(source_path)
    with Image.open(source_path) as opened:
        source = opened.convert("RGBA")
    backend = QwenImageEditReferenceBackgroundInpainter(
        QwenImageEditBenchmarkConfig(model_path=model_path)
    )

    backend.generate(
        source_rgba=source,
        entity_phrase="a person",
        reference_type="subject",
        seed=0,
        prompt="generate background",
    )

    assert events[0][0] == "load"
    assert events[0][3] is True
    assert not any(event == "load_lora" for event in events)


def test_constant_background_fails_hard_check(environment: _Environment) -> None:
    backend = _Backend({0: Image.new("RGB", (11, 7), (255, 255, 255))})
    judge = _Judge([_accept()])
    stats = run_reference_inpaint_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        backend=backend,
        judge=judge,
        seeds=(0,),
    )

    result = json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )
    assert stats.rejected == 1
    assert result["attempts"][0]["hard_check"]["status"] == "failed"
    assert (
        "generated_background_is_constant"
        in result["attempts"][0]["hard_check"]["reasons"]
    )
    assert judge.calls == []


def test_first_accepted_seed_wins_while_all_candidates_are_retained(
    environment: _Environment,
) -> None:
    backend = _Backend({0: _gradient((11, 7)), 17: _gradient((11, 7))})
    _run(environment, backend=backend, judge=_Judge([_accept(), _accept()]))

    result = json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )
    assert [call["seed"] for call in backend.calls] == [0, 17]
    assert result["accepted_candidate"]["seed"] == 0


def test_first_reject_then_second_seed_runs_in_order(
    environment: _Environment,
) -> None:
    backend = _Backend({0: _gradient((11, 7)), 17: _gradient((11, 7))})
    judge = _Judge([_reject(), _accept()])
    _run(environment, backend=backend, judge=judge)

    result = json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )
    assert [call["seed"] for call in backend.calls] == [0, 17]
    assert result["accepted_candidate"]["seed"] == 17
    assert [attempt["seed"] for attempt in result["attempts"]] == [0, 17]


def test_all_candidates_rejected_without_source_fallback(
    environment: _Environment,
) -> None:
    backend = _Backend({0: _gradient((11, 7)), 17: _gradient((11, 7))})
    stats = run_reference_inpaint_benchmark(
        manifest_path=environment.manifest,
        benchmark_root=environment.output_root,
        backend=backend,
        judge=_Judge([_reject(), _reject()]),
    )

    result = json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )
    assert stats.to_dict() == {"processed": 1, "accepted": 0, "rejected": 1}
    assert result["status"] == "rejected"
    assert result["accepted_candidate"] is None


def _review_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "verdict": "accept",
        "entity_unchanged": True,
        "no_duplicate_entity": True,
        "no_new_salient_entity": True,
        "background_natural": True,
        "boundary_clean": True,
        "reference_usable": True,
        "reason": "All checks pass.",
    }
    payload.update(updates)
    return payload


def test_review_schema_is_strict_and_verdict_is_deterministic() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        ReferenceBackgroundReview.model_validate(_review_payload(extra="forbidden"))
    with pytest.raises(ValidationError, match="if and only if"):
        ReferenceBackgroundReview.model_validate(_review_payload(verdict="reject"))


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
) -> QwenReferenceBackgroundJudge:
    return QwenReferenceBackgroundJudge(
        QwenServiceConfig(model="local-qwen-vl", max_tokens=500),
        repair_retries=repair_retries,
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )


def _review_images() -> tuple[Image.Image, Image.Image, Image.Image]:
    source_array = np.zeros((7, 11, 4), dtype=np.uint8)
    source_array[2:6, 3:8, :3] = (190, 80, 35)
    source_array[2:6, 3:8, 3] = 255
    source = Image.fromarray(source_array)
    return source, _gradient((11, 7)), source.getchannel("A")


def _review_with_qwen(
    judge: QwenReferenceBackgroundJudge,
) -> ReferenceBackgroundReview:
    source, candidate, mask = _review_images()
    return judge.review(
        source_rgba=source,
        candidate_rgb=candidate,
        alpha_mask=mask,
        entity_phrase="a red-coated person",
        reference_type="subject",
    )


def test_qwen_judge_requests_strict_json_schema() -> None:
    completions = _Completions([_review_payload()])

    review = _review_with_qwen(_qwen_judge(completions))

    assert review.verdict == "accept"
    response_format = completions.calls[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True


def test_qwen_judge_falls_back_to_json_object() -> None:
    completions = _Completions(
        [_review_payload()],
        strict_failure=True,
    )

    review = _review_with_qwen(_qwen_judge(completions, repair_retries=0))

    assert review.verdict == "accept"
    assert [call["response_format"]["type"] for call in completions.calls] == [
        "json_schema",
        "json_object",
    ]


def test_qwen_judge_repairs_malformed_structured_output() -> None:
    completions = _Completions(["not json", _review_payload()])

    review = _review_with_qwen(_qwen_judge(completions))

    assert review.verdict == "accept"
    repair_text = completions.calls[1]["messages"][1]["content"][0]["text"]
    assert "Original invalid response" in repair_text
    assert "not json" in repair_text


def test_malformed_qwen_review_fails_closed() -> None:
    completions = _Completions(["not json", "still not json"])

    with pytest.raises(ReferenceBackgroundJudgeFailure):
        _review_with_qwen(_qwen_judge(completions))


def test_sample_publication_is_transactional(environment: _Environment) -> None:
    backend = _Backend({0: AssertionError("stop before publication")})

    with pytest.raises(AssertionError, match="stop before publication"):
        run_reference_inpaint_benchmark(
            manifest_path=environment.manifest,
            benchmark_root=environment.output_root,
            backend=backend,
            judge=_Judge([_accept()]),
            seeds=(0,),
        )

    assert not (environment.output_root / "sample-1").exists()
    assert list(environment.output_root.glob(".sample-1.tmp-*")) == []


def test_benchmark_does_not_modify_source(environment: _Environment) -> None:
    before = environment.source.read_bytes()
    before_sha = hashlib.sha256(before).hexdigest()

    _run(environment)

    assert environment.source.read_bytes() == before
    result = json.loads(
        (environment.output_root / "sample-1" / "result.json").read_text()
    )
    assert result["source_sha256"] == before_sha


def test_benchmark_does_not_touch_production_state(
    environment: _Environment,
) -> None:
    sentinels = []
    for relative in (
        "r2v_v3_runs/run-1/clips/clip-1/clip.json",
        "r2v_v3_datasets/dataset-1/pairing.json",
        "selected/reference.png",
    ):
        path = environment.data_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        sentinels.append((path, path.read_bytes()))

    _run(environment)

    assert all(path.read_bytes() == value for path, value in sentinels)
    assert "reference_inpaint_benchmark" not in STAGE_ORDER


def test_source_path_escape_is_rejected(
    environment: _Environment,
    tmp_path: Path,
) -> None:
    escaped = (tmp_path / "outside.png").resolve()
    _write_source(escaped)
    _write_manifest(environment.manifest, escaped)

    with pytest.raises(ValueError, match="source_rgba_path must remain under"):
        _run(environment)


@pytest.mark.parametrize("relative", ["../outside", "selected/run-1"])
def test_output_path_escape_or_selected_is_rejected(
    environment: _Environment,
    relative: str,
) -> None:
    output = environment.benchmark_base / relative

    with pytest.raises(ValueError, match="strictly below|selected"):
        run_reference_inpaint_benchmark(
            manifest_path=environment.manifest,
            benchmark_root=output,
            backend=_Backend({0: _gradient((11, 7))}),
            judge=_Judge([_accept()]),
            seeds=(0,),
        )


def test_fake_backend_and_judge_publish_complete_offline_result(
    environment: _Environment,
) -> None:
    backend, judge = _run(environment)
    sample = environment.output_root / "sample-1"

    assert len(backend.calls) == 2
    assert len(judge.calls) == 2
    assert sorted(path.name for path in sample.iterdir()) == [
        "baseline_white.png",
        "candidate_seed_0.png",
        "candidate_seed_17.png",
        "result.json",
        "review_seed_0.json",
        "review_seed_17.json",
        "source_rgba.png",
    ]
    result = json.loads((sample / "result.json").read_text())
    assert result["status"] == "accepted"
    assert result["accepted_candidate"]["candidate_path"] == ("candidate_seed_0.png")
    candidate = sample / "candidate_seed_0.png"
    assert result["accepted_candidate"]["candidate_sha256"] == (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
    )


def test_prompt_contains_simple_background_and_entity_preservation_contract() -> None:
    prompt = build_reference_background_prompt(
        entity_phrase="a blue ceramic vase",
        reference_type="object",
    )

    for fragment in (
        "clean, natural, non-salient background",
        "Preserve the supplied entity exactly",
        "transparent background",
        "Do not add another person, animal, vehicle, product, text, sign",
        "Do not change its pose, clothing, anatomy, geometry, texture, or color",
    ):
        assert fragment in prompt
