from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.config import PipelineConfig, _qwen_services, load_config
from r2v_data_v2.inpainting import (
    BACKGROUND_INPAINT_PROMPT,
    BackgroundPromptGenerationError,
    Flux1FillBackend,
    ProductionConsistencyValidator,
    QwenBackgroundFillPromptGenerator,
    _background_pixel_diagnostics,
    _hard_composite,
    _inpainting_prompt,
    _read_mask,
    _repair_mask_for_reference,
    _resolve_background_fill_prompt,
    _save_lossless_atomic,
)


def _is_at_or_below(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _reference_artifacts(
    config: PipelineConfig,
    clip_uids: set[str] | None,
) -> list[tuple[Path, dict[str, object]]]:
    artifacts = sorted(
        [
            *config.output_root.glob("references/*/*/metadata.json"),
            *config.output_root.glob(
                "references/*/*/reference_metadata.json"
            ),
        ]
    )
    selected = []
    for artifact in artifacts:
        value = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            continue
        if value.get("reference_type") != "background":
            continue
        if not value.get("needs_inpainting", False):
            continue
        if clip_uids and str(value.get("clip_uid")) not in clip_uids:
            continue
        selected.append((artifact, value))
    return selected


def _append_jsonl(path: Path, value: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def run_benchmark(
    config: PipelineConfig,
    *,
    output_dir: Path,
    prompt_modes: list[str],
    guidance_scales: list[float],
    steps: list[int],
    seeds: list[int],
    clip_uids: set[str] | None = None,
) -> Path:
    production_root = config.output_root.expanduser().resolve(strict=False)
    requested_root = output_dir.expanduser().resolve(strict=False)
    if _is_at_or_below(requested_root, production_root):
        raise ValueError(
            "benchmark output must be outside the production output_root"
        )
    timestamp = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + f"_{time.time_ns() % 1_000_000_000:09d}"
    )
    run_root = requested_root / f"flux_background_fill_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    jsonl_path = run_root / "candidates.jsonl"
    rows: list[dict[str, object]] = []
    repair_config = _qwen_services(config.qwen).repair_judge
    prompt_generator = (
        QwenBackgroundFillPromptGenerator(repair_config)
        if repair_config is not None
        else None
    )
    flux_backend = Flux1FillBackend(config.inpainting)
    validator = ProductionConsistencyValidator(config)

    for reference_index, (artifact, reference) in enumerate(
        _reference_artifacts(config, clip_uids)
    ):
        raw_path = Path(
            str(
                reference.get("raw_canonical_path")
                or reference.get("canonical_path")
            )
        )
        mask_path = Path(str(reference["mask_path"]))
        original_image = Image.open(raw_path).convert("RGB")
        original = np.asarray(original_image)
        (
            generation_mask,
            mode,
            preflight_reasons,
            source_ratio,
            generation_ratio,
        ) = _repair_mask_for_reference(
            reference,
            raw_shape=original.shape[:2],
            config=config.inpainting,
        )
        mask_preflight_failed = (
            mode != "background_hole_fill"
            or generation_mask is None
            or not generation_mask.any()
            or bool(preflight_reasons)
        )
        generation_mask_image = (
            None
            if generation_mask is None
            else Image.fromarray(generation_mask.astype(np.uint8) * 255)
        )
        prompt_cache: dict[
            str,
            tuple[
                str | None,
                str | None,
                dict[str, object] | None,
                Exception | None,
            ],
        ] = {}
        for prompt_mode in prompt_modes:
            if prompt_mode not in prompt_cache:
                fill_prompt: str | None = None
                production_prompt: str | None = None
                prompt_metadata: dict[str, object] | None = None
                prompt_error: Exception | None = None
                if not mask_preflight_failed:
                    assert generation_mask_image is not None
                    prompt_cache_dir = (
                        run_root
                        / "prompt_cache"
                        / f"reference_{reference_index:05d}"
                        / prompt_mode
                    )
                    prompt_cache_dir.mkdir(parents=True, exist_ok=True)
                    cached_generation_mask_path = (
                        prompt_cache_dir / "generation_mask.png"
                    )
                    _save_lossless_atomic(
                        generation_mask_image,
                        cached_generation_mask_path,
                    )
                    try:
                        if prompt_mode == "empty":
                            fill_prompt = ""
                            production_prompt = ""
                            prompt_metadata = {
                                "source": "empty",
                                "prompt_mode": "empty",
                                "context_image_path": None,
                                "context_image_sha256": None,
                            }
                        elif prompt_mode == "generic":
                            fill_prompt = BACKGROUND_INPAINT_PROMPT
                            production_prompt = _inpainting_prompt(
                                reference,
                                mode,
                                background_fill_prompt=fill_prompt,
                            )
                            prompt_metadata = {
                                "source": "generic",
                                "prompt_mode": "generic",
                                "context_image_path": None,
                                "context_image_sha256": None,
                            }
                        elif prompt_mode == "qwen_local":
                            prompt_config = replace(
                                config,
                                inpainting=replace(
                                    config.inpainting,
                                    background=replace(
                                        config.inpainting.background,
                                        prompt_mode=(
                                            "qwen_local_background"
                                        ),
                                    ),
                                ),
                            )
                            fill_prompt, prompt_metadata = (
                                _resolve_background_fill_prompt(
                                    config=prompt_config,
                                    reference=reference,
                                    original_path=raw_path,
                                    generation_mask_path=(
                                        cached_generation_mask_path
                                    ),
                                    generator=prompt_generator,
                                )
                            )
                            production_prompt = _inpainting_prompt(
                                reference,
                                mode,
                                background_fill_prompt=fill_prompt,
                            )
                        else:
                            raise ValueError(
                                f"unsupported prompt mode: {prompt_mode}"
                            )
                    except Exception as exc:  # noqa: BLE001
                        prompt_error = exc
                prompt_cache[prompt_mode] = (
                    fill_prompt,
                    production_prompt,
                    prompt_metadata,
                    prompt_error,
                )
            (
                fill_prompt,
                production_prompt,
                prompt_metadata,
                prompt_error,
            ) = prompt_cache[prompt_mode]
            for guidance_scale in guidance_scales:
                for inference_steps in steps:
                    for seed in seeds:
                        index = len(rows)
                        candidate_dir = run_root / f"candidate_{index:05d}"
                        candidate_dir.mkdir()
                        local_reference = dict(reference)
                        local_raw_path = candidate_dir / "original.png"
                        _save_lossless_atomic(original_image, local_raw_path)
                        local_reference["raw_canonical_path"] = str(
                            local_raw_path
                        )
                        tuned_inpainting = replace(
                            config.inpainting,
                            guidance_scale=guidance_scale,
                            num_inference_steps=inference_steps,
                        )
                        record: dict[str, object] = {
                            "clip_uid": reference.get("clip_uid"),
                            "reference_id": reference.get("reference_id"),
                            "artifact_path": str(artifact),
                            "prompt_mode": prompt_mode,
                            "guidance_scale": guidance_scale,
                            "num_inference_steps": inference_steps,
                            "seed": seed,
                            "source_foreground_area_ratio": source_ratio,
                            "generation_mask_area_ratio": generation_ratio,
                            "accepted": False,
                            "rejection_reasons": list(preflight_reasons),
                            "candidate_dir": str(candidate_dir),
                        }
                        try:
                            if (
                                mode != "background_hole_fill"
                                or generation_mask is None
                                or not generation_mask.any()
                                or preflight_reasons
                            ):
                                raise ValueError(
                                    "background candidate failed mask preflight"
                                )
                            generation_mask_path = (
                                candidate_dir / "generation_mask.png"
                            )
                            generation_mask_image = Image.fromarray(
                                generation_mask.astype(np.uint8) * 255
                            )
                            _save_lossless_atomic(
                                generation_mask_image,
                                generation_mask_path,
                            )
                            if prompt_error is not None:
                                raise prompt_error
                            assert fill_prompt is not None
                            assert production_prompt is not None
                            assert prompt_metadata is not None
                            generated = flux_backend.inpaint(
                                image=original_image,
                                mask=generation_mask_image,
                                prompt=production_prompt,
                                seed=seed,
                                guidance_scale=guidance_scale,
                                num_inference_steps=inference_steps,
                            )
                            generated_path = candidate_dir / "generated.png"
                            _save_lossless_atomic(generated, generated_path)
                            final, _ = _hard_composite(
                                original=original,
                                generated=np.asarray(generated),
                                core_mask=generation_mask,
                                feather_pixels=(
                                    tuned_inpainting.feather_pixels
                                ),
                            )
                            final_path = candidate_dir / "candidate.png"
                            _save_lossless_atomic(
                                Image.fromarray(final),
                                final_path,
                            )
                            diagnostics = _background_pixel_diagnostics(
                                original=original,
                                repaired=final,
                                source_mask=_read_mask(
                                    mask_path, original.shape[:2]
                                ),
                                generation_mask=generation_mask,
                                destination=(
                                    candidate_dir
                                    / "difference_heatmap.png"
                                ),
                            )
                            validation = validator(
                                original=original_image,
                                repaired=Image.fromarray(final),
                                repair_mask=generation_mask_image,
                                reference=local_reference,
                                mode=mode,
                                diagnostics_dir=candidate_dir,
                            )
                            record.update(
                                {
                                    **diagnostics,
                                    "fill_prompt": fill_prompt,
                                    "production_prompt": production_prompt,
                                    "prompt_source": prompt_metadata["source"],
                                    "prompt_metadata": prompt_metadata,
                                    "prompt_context_image_path": (
                                        prompt_metadata.get(
                                            "context_image_path"
                                        )
                                    ),
                                    "prompt_context_image_sha256": (
                                        prompt_metadata.get(
                                            "context_image_sha256"
                                        )
                                    ),
                                    "candidate_path": str(final_path),
                                    "validator": validation,
                                    "accepted": (
                                        validation.get("accepted") is True
                                    ),
                                    "rejection_reasons": validation.get(
                                        "rejection_reasons", []
                                    ),
                                }
                            )
                        except BackgroundPromptGenerationError as exc:
                            record.update(
                                {
                                    "prompt_source": (
                                        "qwen_local_background"
                                    ),
                                    "prompt_metadata": exc.metadata,
                                    "prompt_context_image_path": (
                                        exc.metadata.get(
                                            "context_image_path"
                                        )
                                    ),
                                    "prompt_context_image_sha256": (
                                        exc.metadata.get(
                                            "context_image_sha256"
                                        )
                                    ),
                                    "rejection_reasons": [
                                        "background_prompt_generation_failed"
                                    ],
                                    "error": str(exc),
                                }
                            )
                        except Exception as exc:  # noqa: BLE001
                            record["error"] = str(exc)
                            if not record["rejection_reasons"]:
                                record["rejection_reasons"] = [
                                    "benchmark_candidate_failed"
                                ]
                        rows.append(record)
                        _append_jsonl(jsonl_path, record)
    validator.close()

    columns = (
        "clip_uid",
        "reference_id",
        "prompt_mode",
        "prompt_source",
        "prompt_context_image_path",
        "prompt_context_image_sha256",
        "guidance_scale",
        "num_inference_steps",
        "seed",
        "accepted",
        "source_foreground_area_ratio",
        "generation_mask_area_ratio",
        "masked_mean_l1",
        "masked_changed_pixel_ratio",
        "generation_mask_changed_pixel_ratio",
        "inner_boundary_mean_l1",
        "outer_unmasked_boundary_mean_l1",
        "candidate_dir",
        "error",
    )
    with (run_root / "candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})
    summary = {
        "candidate_count": len(rows),
        "accepted_count": sum(row["accepted"] is True for row in rows),
        "production_output_root": str(production_root),
        "benchmark_output_root": str(run_root),
        "prompt_modes": prompt_modes,
        "guidance_scales": guidance_scales,
        "num_inference_steps": steps,
        "seeds": seeds,
        "clip_uids": sorted(clip_uids) if clip_uids else None,
    }
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FLUX background fill without mutating production"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-modes",
        nargs="+",
        choices=("empty", "generic", "qwen_local"),
        default=["empty", "generic", "qwen_local"],
    )
    parser.add_argument(
        "--guidance-scales",
        nargs="+",
        type=float,
        default=[20.0, 30.0],
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[50],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 17],
    )
    parser.add_argument("--clip-uids", nargs="*")
    args = parser.parse_args()
    destination = run_benchmark(
        load_config(args.config),
        output_dir=args.output_dir,
        prompt_modes=args.prompt_modes,
        guidance_scales=args.guidance_scales,
        steps=args.steps,
        seeds=args.seeds,
        clip_uids=set(args.clip_uids) if args.clip_uids else None,
    )
    print(destination)


if __name__ == "__main__":
    main()
