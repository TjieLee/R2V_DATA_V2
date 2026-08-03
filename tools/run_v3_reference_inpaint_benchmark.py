from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.reference_inpaint_benchmark import (
    DEFAULT_SEEDS,
    QwenImageEditBenchmarkConfig,
    QwenImageEditReferenceBackgroundInpainter,
    QwenReferenceBackgroundJudge,
    run_reference_inpaint_benchmark,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the offline V3 entity-reference background-inpainting benchmark"
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
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
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--true-cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Use once or twice; defaults to seeds 0 and 17.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, int]:
    arguments = _parser().parse_args(argv)
    backend = QwenImageEditReferenceBackgroundInpainter(
        QwenImageEditBenchmarkConfig(
            model_path=arguments.model_path,
            device=arguments.device,
            dtype=arguments.dtype,
            num_inference_steps=arguments.num_inference_steps,
            true_cfg_scale=arguments.true_cfg_scale,
            guidance_scale=arguments.guidance_scale,
        )
    )
    judge = QwenReferenceBackgroundJudge(
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
        stats = run_reference_inpaint_benchmark(
            manifest_path=arguments.manifest,
            benchmark_root=arguments.benchmark_root,
            backend=backend,
            judge=judge,
            seeds=tuple(arguments.seeds or DEFAULT_SEEDS),
        )
    finally:
        backend.close()
        judge.close()
    result = stats.to_dict()
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
