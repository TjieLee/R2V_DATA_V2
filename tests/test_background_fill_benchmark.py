from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from r2v_data_v2.config import (
    InpaintingBackgroundConfig,
    InpaintingConfig,
    PipelineConfig,
    QwenConfig,
    QwenServicesConfig,
)
from r2v_data_v2.schemas import BackgroundFillPrompt
from scripts import benchmark_flux_background_fill as benchmark


def _write_background_reference(output_root: Path) -> None:
    destination = output_root / "references" / "clip-1" / "bg1"
    destination.mkdir(parents=True)
    raw_path = destination / "canonical_raw.jpg"
    Image.new("RGB", (32, 32), (80, 90, 100)).save(raw_path)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:16, 12:16] = 255
    mask_path = destination / "foreground_mask.png"
    Image.fromarray(mask).save(mask_path)
    (destination / "reference_metadata.json").write_text(
        json.dumps(
            {
                "clip_uid": "clip-1",
                "reference_id": "bg1",
                "reference_type": "background",
                "phrase": "a stone courtyard",
                "raw_canonical_path": str(raw_path),
                "canonical_path": str(raw_path),
                "mask_path": str(mask_path),
                "needs_inpainting": True,
                "status": "pending_inpainting",
            }
        ),
        encoding="utf-8",
    )


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_background_fill_benchmark_never_mutates_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "production"
    _write_background_reference(output_root)
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=output_root,
        inpainting=InpaintingConfig(
            enabled=True,
            backend="noop",
            mask_dilation_pixels=1,
            background=InpaintingBackgroundConfig(
                prompt_mode="generic",
                candidate_seeds=[0],
            ),
        ),
    )
    before = _tree_snapshot(output_root)
    backend_instances: list[object] = []
    backend_calls: list[tuple[float, int, str]] = []

    class _Backend:
        def __init__(self, config: InpaintingConfig) -> None:
            del config
            backend_instances.append(self)

        def inpaint(self, **kwargs: object) -> Image.Image:
            image = kwargs["image"]
            assert isinstance(image, Image.Image)
            backend_calls.append(
                (
                    float(kwargs["guidance_scale"]),
                    int(kwargs["num_inference_steps"]),
                    str(kwargs["prompt"]),
                )
            )
            return Image.new("RGB", image.size, (120, 130, 140))

    class _Validator:
        def __init__(self, config: PipelineConfig) -> None:
            del config

        def __call__(self, **kwargs: object) -> dict[str, object]:
            diagnostics_dir = kwargs["diagnostics_dir"]
            assert isinstance(diagnostics_dir, Path)
            return {
                "accepted": True,
                "rejection_reasons": [],
                "comparison_sheet_paths": [],
            }

        def close(self) -> None:
            return None

    monkeypatch.setattr(benchmark, "Flux1FillBackend", _Backend)
    monkeypatch.setattr(
        benchmark,
        "ProductionConsistencyValidator",
        _Validator,
    )
    run_root = benchmark.run_benchmark(
        config,
        output_dir=tmp_path / "benchmarks",
        prompt_modes=["generic"],
        guidance_scales=[20.0, 30.0],
        steps=[28, 50],
        seeds=[0],
    )

    assert _tree_snapshot(output_root) == before
    assert len(backend_instances) == 1
    assert {
        (guidance, steps)
        for guidance, steps, _ in backend_calls
    } == {(20.0, 28), (20.0, 50), (30.0, 28), (30.0, 50)}
    assert all(prompt for _, _, prompt in backend_calls)
    assert (run_root / "candidates.jsonl").is_file()
    assert (run_root / "candidates.csv").is_file()
    summary = json.loads(
        (run_root / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["candidate_count"] == 4
    assert summary["accepted_count"] == 4


def test_background_fill_benchmark_empty_mode_sends_empty_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "production"
    _write_background_reference(output_root)
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=output_root,
        inpainting=InpaintingConfig(
            enabled=True,
            backend="noop",
            mask_dilation_pixels=1,
            background=InpaintingBackgroundConfig(
                prompt_mode="generic",
                candidate_seeds=[0],
            ),
        ),
    )
    prompts: list[str] = []

    class _Backend:
        def __init__(self, config: InpaintingConfig) -> None:
            del config

        def inpaint(self, **kwargs: object) -> Image.Image:
            image = kwargs["image"]
            assert isinstance(image, Image.Image)
            prompts.append(str(kwargs["prompt"]))
            return image.copy()

    class _Validator:
        def __init__(self, config: PipelineConfig) -> None:
            del config

        def __call__(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {"accepted": True, "rejection_reasons": []}

        def close(self) -> None:
            return None

    monkeypatch.setattr(benchmark, "Flux1FillBackend", _Backend)
    monkeypatch.setattr(
        benchmark,
        "ProductionConsistencyValidator",
        _Validator,
    )
    run_root = benchmark.run_benchmark(
        config,
        output_dir=tmp_path / "benchmarks",
        prompt_modes=["empty"],
        guidance_scales=[20.0],
        steps=[50],
        seeds=[0],
    )
    record = json.loads(
        (run_root / "candidates.jsonl").read_text(encoding="utf-8")
    )

    assert prompts == [""]
    assert record["prompt_mode"] == "empty"
    assert record["prompt_source"] == "empty"
    assert record["fill_prompt"] == ""
    assert record["production_prompt"] == ""


def test_background_fill_benchmark_qwen_failure_is_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "production"
    _write_background_reference(output_root)
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=output_root,
        inpainting=InpaintingConfig(
            enabled=True,
            backend="noop",
            mask_dilation_pixels=1,
            background=InpaintingBackgroundConfig(
                prompt_mode="qwen_local_background",
                candidate_seeds=[0],
            ),
        ),
    )

    class _Backend:
        call_count = 0

        def __init__(self, config: InpaintingConfig) -> None:
            del config

        def inpaint(self, **kwargs: object) -> Image.Image:
            del kwargs
            self.__class__.call_count += 1
            raise AssertionError("FLUX must not run after Qwen prompt failure")

    monkeypatch.setattr(benchmark, "Flux1FillBackend", _Backend)
    run_root = benchmark.run_benchmark(
        config,
        output_dir=tmp_path / "benchmarks",
        prompt_modes=["qwen_local"],
        guidance_scales=[20.0],
        steps=[50],
        seeds=[0],
    )
    record = json.loads(
        (run_root / "candidates.jsonl").read_text(encoding="utf-8")
    )

    assert _Backend.call_count == 0
    assert record["accepted"] is False
    assert record["prompt_mode"] == "qwen_local"
    assert record["prompt_source"] == "qwen_local_background"
    assert record["rejection_reasons"] == [
        "background_prompt_generation_failed"
    ]
    assert "generic" not in json.dumps(record).casefold()


def test_background_fill_benchmark_caches_qwen_prompt_across_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "production"
    _write_background_reference(output_root)
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=output_root,
        qwen=QwenServicesConfig(repair_judge=QwenConfig(model="fake-qwen")),
        inpainting=InpaintingConfig(
            enabled=True,
            backend="noop",
            mask_dilation_pixels=1,
            background=InpaintingBackgroundConfig(
                prompt_mode="qwen_local_background",
                candidate_seeds=[0],
            ),
        ),
    )
    generator_inputs: list[tuple[Path, Path]] = []
    flux_prompts: list[str] = []

    class _PromptGenerator:
        def __init__(self, config: QwenConfig) -> None:
            assert config.model == "fake-qwen"

        def generate(
            self,
            *,
            original_path: Path,
            generation_mask_path: Path,
            forbidden_texts: list[str],
        ) -> tuple[BackgroundFillPrompt, dict[str, object]]:
            assert forbidden_texts
            generator_inputs.append((original_path, generation_mask_path))
            return (
                BackgroundFillPrompt(
                    fill_prompt=(
                        "Weathered stone paving continues naturally with "
                        "consistent gray texture, perspective, and daylight"
                    ),
                    visible_background_elements=["stone paving"],
                    reason="Stone paving surrounds the masked area.",
                ),
                {"model": "fake-qwen", "validation": "passed"},
            )

    class _Backend:
        def __init__(self, config: InpaintingConfig) -> None:
            del config

        def inpaint(self, **kwargs: object) -> Image.Image:
            image = kwargs["image"]
            assert isinstance(image, Image.Image)
            flux_prompts.append(str(kwargs["prompt"]))
            return image.copy()

    class _Validator:
        def __init__(self, config: PipelineConfig) -> None:
            del config

        def __call__(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {"accepted": True, "rejection_reasons": []}

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        benchmark,
        "QwenBackgroundFillPromptGenerator",
        _PromptGenerator,
    )
    monkeypatch.setattr(benchmark, "Flux1FillBackend", _Backend)
    monkeypatch.setattr(
        benchmark,
        "ProductionConsistencyValidator",
        _Validator,
    )

    run_root = benchmark.run_benchmark(
        config,
        output_dir=tmp_path / "benchmarks",
        prompt_modes=["qwen_local"],
        guidance_scales=[20.0, 30.0],
        steps=[28, 50],
        seeds=[0, 17],
    )
    records = [
        json.loads(line)
        for line in (run_root / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(generator_inputs) == 1
    assert len(flux_prompts) == 8
    assert len(set(flux_prompts)) == 1
    assert len(records) == 8
    assert len({record["fill_prompt"] for record in records}) == 1
    assert len({record["production_prompt"] for record in records}) == 1
    assert all(
        record["prompt_metadata"] == records[0]["prompt_metadata"]
        for record in records
    )
    assert {
        record["prompt_context_image_path"] for record in records
    } == {str(generator_inputs[0][0])}
    assert len(
        {
            record["prompt_context_image_sha256"]
            for record in records
        }
    ) == 1


def test_background_fill_benchmark_rejects_production_output_path(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        dataset_json=tmp_path / "source.jsonl",
        output_root=tmp_path / "production",
    )

    with pytest.raises(ValueError, match="outside the production"):
        benchmark.run_benchmark(
            config,
            output_dir=config.output_root / "benchmark",
            prompt_modes=["generic"],
            guidance_scales=[20.0],
            steps=[50],
            seeds=[0],
        )
