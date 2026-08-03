from __future__ import annotations

import argparse
import json
from pathlib import Path

from r2v_data_v2.v3.config import QwenServiceConfig
from r2v_data_v2.v3.reference_completion_publish import (
    DEFAULT_PUBLICATION_ROOT,
    DEFAULT_SAM3_CHECKPOINT,
    DEFAULT_SAM3_CODE_ROOT,
    PublicationConfig,
    QwenCompletionPublicationJudge,
    Sam3LocalizedCompletionSegmenter,
    run_reference_completion_publication,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run fail-closed automatic publication gates for V3 Qwen "
            "localized completion artifacts"
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sam3-code-root", type=Path, default=DEFAULT_SAM3_CODE_ROOT)
    parser.add_argument(
        "--sam3-checkpoint",
        type=Path,
        default=DEFAULT_SAM3_CHECKPOINT,
    )
    parser.add_argument("--sam3-device", default="cuda")
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-timeout-seconds", type=int, default=3600)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument("--repair-retries", type=int, default=1)
    parser.add_argument("--largest-component-ratio", type=float, default=0.90)
    parser.add_argument(
        "--max-secondary-component-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--group-min-largest-component-ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--group-max-secondary-component-ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--group-significant-component-ratio",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--group-max-component-gap-ratio",
        type=float,
        default=0.30,
    )
    parser.add_argument("--min-area-ratio-vs-source", type=float, default=0.75)
    parser.add_argument("--max-area-ratio-vs-source", type=float, default=2.00)
    parser.add_argument(
        "--group-min-area-ratio-vs-source",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--group-max-area-ratio-vs-source",
        type=float,
        default=2.50,
    )
    parser.add_argument("--min-border-mean", type=float, default=235.0)
    parser.add_argument("--max-border-std", type=float, default=30.0)
    parser.add_argument("--near-white-channel-min", type=int, default=225)
    parser.add_argument(
        "--min-outside-near-white-ratio",
        type=float,
        default=0.80,
    )
    parser.add_argument("--max-outside-color-bins", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> dict[str, int]:
    arguments = _parser().parse_args(argv)
    if not arguments.run_id.strip():
        raise ValueError("run-id must be non-empty")
    config = PublicationConfig(
        largest_component_ratio=arguments.largest_component_ratio,
        max_secondary_component_ratio=(
            arguments.max_secondary_component_ratio
        ),
        group_min_largest_component_ratio=(
            arguments.group_min_largest_component_ratio
        ),
        group_max_secondary_component_ratio=(
            arguments.group_max_secondary_component_ratio
        ),
        group_significant_component_ratio=(
            arguments.group_significant_component_ratio
        ),
        group_max_component_gap_ratio=(
            arguments.group_max_component_gap_ratio
        ),
        min_area_ratio_vs_source=arguments.min_area_ratio_vs_source,
        max_area_ratio_vs_source=arguments.max_area_ratio_vs_source,
        group_min_area_ratio_vs_source=(
            arguments.group_min_area_ratio_vs_source
        ),
        group_max_area_ratio_vs_source=(
            arguments.group_max_area_ratio_vs_source
        ),
        min_border_mean=arguments.min_border_mean,
        max_border_std=arguments.max_border_std,
        near_white_channel_min=arguments.near_white_channel_min,
        min_outside_near_white_ratio=(
            arguments.min_outside_near_white_ratio
        ),
        max_outside_color_bins=arguments.max_outside_color_bins,
    )
    service = QwenServiceConfig(
        base_url=arguments.judge_base_url,
        api_key=arguments.judge_api_key,
        model=arguments.judge_model,
        temperature=0.0,
        max_tokens=arguments.judge_max_tokens,
        timeout_seconds=arguments.judge_timeout_seconds,
    )
    segmenter = Sam3LocalizedCompletionSegmenter(
        code_root=arguments.sam3_code_root,
        checkpoint_path=arguments.sam3_checkpoint,
        device=arguments.sam3_device,
    )
    judges = {
        kind: QwenCompletionPublicationJudge(
            service,
            kind=kind,
            repair_retries=arguments.repair_retries,
        )
        for kind in ("identity", "locality", "usability")
    }
    try:
        stats = run_reference_completion_publication(
            manifest_path=arguments.manifest,
            benchmark_root=DEFAULT_PUBLICATION_ROOT / arguments.run_id,
            config=config,
            segmenter=segmenter,
            identity_judge=judges["identity"],
            locality_judge=judges["locality"],
            usability_judge=judges["usability"],
        )
    finally:
        segmenter.close()
        for judge in judges.values():
            judge.close()
    result = stats.to_dict()
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
