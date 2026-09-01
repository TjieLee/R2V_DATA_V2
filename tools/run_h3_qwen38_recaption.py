#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.jea_audio_production import jea_production_paths
from r2v_data_v2.h3.qwen38_h3_recaption import (
    DEFAULT_CHECKPOINT_ID,
    DEFAULT_MODEL,
    OpenAIQwen38RecaptionBackend,
    Qwen38RecaptionConfig,
    build_qwen38_full_manifest,
    build_qwen38_pilot_manifest,
    run_qwen38_h3_recaption_pilot,
)
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the read-only Qwen3.8 MiniMax H3 recaption pilot",
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--audio-semantics-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prepare-manifest", type=Path)
    parser.add_argument("--prepare-all-manifest", type=Path)
    parser.add_argument("--manifest-size", type=int, default=5)
    parser.add_argument(
        "--conditioning-policy",
        choices=("sample_pair_type",),
    )
    parser.add_argument(
        "--conditioning-variant",
        choices=(
            "visual_only",
            "target_voice_reference",
            "cross_voice_reference",
            "full_audio_reuse",
        ),
        default="visual_only",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN38_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument("--api-key", default=os.environ.get("QWEN38_API_KEY", "EMPTY"))
    parser.add_argument("--served-model-name", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-id", default=DEFAULT_CHECKPOINT_ID)
    parser.add_argument("--media-mode", choices=("file", "http"), default="file")
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--media-base-url")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    paths = jea_production_paths(arguments.audio_production_root)
    preparation_modes = sum(
        value is not None
        for value in (arguments.prepare_manifest, arguments.prepare_all_manifest)
    )
    if preparation_modes > 1:
        raise ValueError(
            "--prepare-manifest and --prepare-all-manifest are mutually exclusive"
        )
    if arguments.prepare_all_manifest is not None:
        if arguments.case_manifest is not None:
            raise ValueError(
                "--prepare-all-manifest and --case-manifest are mutually exclusive"
            )
        if arguments.conditioning_policy != "sample_pair_type":
            raise ValueError(
                "--prepare-all-manifest requires --conditioning-policy sample_pair_type"
            )
        cases = build_qwen38_full_manifest(
            h3_samples_path=paths.h3 / "samples.jsonl",
            output_path=arguments.prepare_all_manifest,
            conditioning_policy=arguments.conditioning_policy,
        )
        result = {
            "manifest_path": str(
                arguments.prepare_all_manifest.expanduser().resolve()
            ),
            "case_count": len(cases),
            "inventory_scope": "current_h3_samples_inventory_only",
            "canonical_wide_coverage": False,
            "model_loaded": False,
            "model_calls": 0,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result
    if arguments.prepare_manifest is not None:
        if arguments.case_manifest is not None:
            raise ValueError("--prepare-manifest and --case-manifest are mutually exclusive")
        cases = build_qwen38_pilot_manifest(
            h3_samples_path=paths.h3 / "samples.jsonl",
            output_path=arguments.prepare_manifest,
            size=arguments.manifest_size,
            conditioning_variant=arguments.conditioning_variant,
        )
        result = {
            "manifest_path": str(arguments.prepare_manifest.expanduser().resolve()),
            "case_count": len(cases),
            "model_loaded": False,
            "model_calls": 0,
        }
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result
    if arguments.case_manifest is None:
        raise ValueError(
            "--case-manifest is required unless a manifest preparation mode is used"
        )
    if arguments.media_root is None:
        raise ValueError("--media-root is required for recaption inference")
    resolver = MediaURLResolver(
        mode=arguments.media_mode,
        media_root=arguments.media_root,
        media_base_url=arguments.media_base_url,
    )
    backend = OpenAIQwen38RecaptionBackend(
        Qwen38RecaptionConfig(
            base_url=arguments.base_url,
            media_resolver=resolver,
            api_key=arguments.api_key,
            served_model_name=arguments.served_model_name,
            checkpoint_id=arguments.checkpoint_id,
            timeout_seconds=arguments.timeout_seconds,
            max_tokens=arguments.max_tokens,
            temperature=arguments.temperature,
            top_p=arguments.top_p,
            top_k=arguments.top_k,
            min_p=arguments.min_p,
            presence_penalty=arguments.presence_penalty,
            repetition_penalty=arguments.repetition_penalty,
        )
    )
    summary = run_qwen38_h3_recaption_pilot(
        audio_production_root=arguments.audio_production_root,
        case_manifest_path=arguments.case_manifest,
        backend=backend,
        audio_semantics_root=arguments.audio_semantics_root,
        output_root=arguments.output_root,
        overwrite=arguments.overwrite,
    )
    result = summary.model_dump(mode="json")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
