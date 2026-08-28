"""Launch the fixed standalone entity-mask Stage2 production job."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3.config import V3Config
from r2v_data_v2.v3.pre_qwen_production import (
    _safe_output_root,
    load_config_identity,
    validate_stage2_preflight,
)
from tools.run_v3_pre_qwen_auto import main as run_pre_qwen_auto

PRODUCTION_INPUT_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/entity_annotations"
)
PRODUCTION_OUTPUT_ROOT = Path(
    "/mnt/workspace/public/dataset/jea-video/"
    "moive-183t-0808_processed/entity_mask"
)
PRODUCTION_BASE_CONFIG = Path(
    "/mnt/workspace/litengjie/data/entity_mask_configs/production.yaml"
)
PRODUCTION_LOG_ROOT = Path(
    "/mnt/workspace/litengjie/data/entity_mask_logs"
)
PRODUCTION_CANDIDATE_JUDGE_BASE_URL = "http://6.167.57.88:8000/v1"
PRODUCTION_CANDIDATE_JUDGE_MODEL = (
    "/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct"
)


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(
            f"entity-mask production requires {name}={expected!r}; got {actual!r}"
        )


def validate_entity_mask_production_config(config: V3Config) -> None:
    validate_stage2_preflight(config)
    service = config.qwen.candidate_judge
    _require_equal("sam3.save_debug_overlays", config.sam3.save_debug_overlays, False)
    _require_equal("debug.save_diagnostics", config.debug.save_diagnostics, False)
    _require_equal(
        "sam3.object_rescue_mode",
        config.sam3.object_rescue_mode,
        "phrase_retry_v1",
    )
    _require_equal(
        "sam3.not_found_rescue_mode",
        config.sam3.not_found_rescue_mode,
        "entity_phrase_retry_v1",
    )
    _require_equal(
        "sam3.multi_instance_rescue_mode",
        config.sam3.multi_instance_rescue_mode,
        "qwen_anchor_select_v1",
    )
    _require_equal(
        "sam3.anchor_search_mode",
        config.sam3.anchor_search_mode,
        "progressive_v1",
    )
    _require_equal(
        "qwen.candidate_judge.base_url",
        service.base_url,
        PRODUCTION_CANDIDATE_JUDGE_BASE_URL,
    )
    _require_equal(
        "qwen.candidate_judge.model",
        service.model,
        PRODUCTION_CANDIDATE_JUDGE_MODEL,
    )
    _require_equal("qwen.candidate_judge.temperature", service.temperature, 0.0)
    _require_equal("qwen.candidate_judge.max_tokens", service.max_tokens, 1024)
    _require_equal(
        "qwen.candidate_judge.timeout_seconds",
        service.timeout_seconds,
        3600,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _production_arguments(*, dry_run: bool) -> list[str]:
    arguments = [
        "--input-root",
        str(PRODUCTION_INPUT_ROOT),
        "--base-config",
        str(PRODUCTION_BASE_CONFIG),
        "--output-root",
        str(PRODUCTION_OUTPUT_ROOT),
        "--chunk-rows",
        "100",
        "--frame-prefetch-workers",
        "0",
        "--sam3-session-reuse-mode",
        "clip_reset_v1",
    ]
    if dry_run:
        arguments.append("--dry-run")
    return arguments


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = _parser().parse_args(argv)
    if not PRODUCTION_BASE_CONFIG.is_file():
        raise FileNotFoundError(
            "missing entity-mask production config: "
            f"{PRODUCTION_BASE_CONFIG}; copy the validated Stage2 config here"
        )
    _safe_output_root(PRODUCTION_OUTPUT_ROOT)
    identity = load_config_identity(PRODUCTION_BASE_CONFIG)
    validate_entity_mask_production_config(identity.config)
    return run_pre_qwen_auto(
        _production_arguments(dry_run=args.dry_run),
        worker_log_root=PRODUCTION_LOG_ROOT,
    )


if __name__ == "__main__":
    main()
