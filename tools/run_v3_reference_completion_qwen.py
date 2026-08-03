from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.reference_completion_benchmark import (
    QwenReferenceCompletionJudge,
)
from r2v_data_v2.v3.reference_completion_qwen import (
    DEFAULT_QWEN_COMPLETION_MODE,
    DEFAULT_QWEN_MODEL_PATH,
    DEFAULT_QWEN_SEEDS,
    LEGACY_QWEN_SEEDS,
    QwenImageEdit2511CompletionConfig,
    QwenImageEdit2511ReferenceCompletionBackend,
    QwenLocalizedReferenceCompletionJudge,
    run_qwen_reference_completion_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline V3 Qwen-Image-Edit-2511 entity-reference "
            "completion benchmark"
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument(
        "--judge-base-url",
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-timeout-seconds", type=int, default=3600)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument("--repair-retries", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the reference-type-specific localized prompt.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Override the mode default; localized_raw defaults to one space.",
    )
    legacy_help = "Used only by legacy whole_canvas / explicit_mask modes."
    parser.add_argument(
        "--canvas-expand-ratio", type=float, default=0.75, help=legacy_help
    )
    parser.add_argument(
        "--lateral-padding-ratio", type=float, default=0.20, help=legacy_help
    )
    parser.add_argument(
        "--mask-overlap-pixels", type=int, default=0, help=legacy_help
    )
    parser.add_argument("--model-min-side", type=int, default=1024, help=legacy_help)
    parser.add_argument("--model-multiple", type=int, default=16, help=legacy_help)
    parser.add_argument(
        "--mode",
        choices=("localized_raw", "whole_canvas", "explicit_mask"),
        default=DEFAULT_QWEN_COMPLETION_MODE,
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help=(
            "Choose and order seeds. localized_raw defaults to 0; legacy modes "
            "retain their 0,17 default."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, int]:
    arguments = _parser().parse_args(argv)
    completion_config = QwenImageEdit2511CompletionConfig(
        model_path=arguments.model_path,
        device=arguments.device,
        dtype=arguments.dtype,
        num_inference_steps=arguments.num_inference_steps,
        true_cfg_scale=arguments.true_cfg_scale,
        guidance_scale=arguments.guidance_scale,
        canvas_expand_ratio=arguments.canvas_expand_ratio,
        lateral_padding_ratio=arguments.lateral_padding_ratio,
        mask_overlap_pixels=arguments.mask_overlap_pixels,
        model_min_side=arguments.model_min_side,
        model_multiple=arguments.model_multiple,
        mode=arguments.mode,
    )
    backend = QwenImageEdit2511ReferenceCompletionBackend(completion_config)
    judge_class = (
        QwenLocalizedReferenceCompletionJudge
        if arguments.mode == "localized_raw"
        else QwenReferenceCompletionJudge
    )
    judge = judge_class(
        QwenServiceConfig(
            base_url=arguments.judge_base_url,
            api_key=arguments.judge_api_key,
            model=arguments.judge_model,
            temperature=0.0,
            max_tokens=arguments.judge_max_tokens,
            timeout_seconds=arguments.judge_timeout_seconds,
        ),
        repair_retries=arguments.repair_retries,
    )
    try:
        default_seeds = (
            DEFAULT_QWEN_SEEDS
            if arguments.mode == "localized_raw"
            else LEGACY_QWEN_SEEDS
        )
        stats = run_qwen_reference_completion_benchmark(
            manifest_path=arguments.manifest,
            benchmark_root=arguments.benchmark_root,
            config=completion_config,
            backend=backend,
            judge=judge,
            prompt=arguments.prompt,
            negative_prompt=arguments.negative_prompt,
            seeds=tuple(arguments.seeds or default_seeds),
        )
    finally:
        backend.close()
        judge.close()
    result = stats.to_dict()
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
