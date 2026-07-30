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
)
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

    class _Backend:
        def __init__(self, config: InpaintingConfig) -> None:
            del config

        def inpaint(self, **kwargs: object) -> Image.Image:
            image = kwargs["image"]
            assert isinstance(image, Image.Image)
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
        guidance_scales=[20.0],
        steps=[50],
        seeds=[0],
    )

    assert _tree_snapshot(output_root) == before
    assert (run_root / "candidates.jsonl").is_file()
    assert (run_root / "candidates.csv").is_file()
    summary = json.loads(
        (run_root / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["candidate_count"] == 1
    assert summary["accepted_count"] == 1


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
