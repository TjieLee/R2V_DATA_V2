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

from r2v_data_v2.h3.jea_target_audio_caption import (
    DEFAULT_QWEN3_OMNI_CHECKPOINT_ID,
    DEFAULT_QWEN3_OMNI_MODEL,
)
from r2v_data_v2.h3.semantic_augmentation import MediaURLResolver
from r2v_data_v2.h3.specialized_audio_semantics import (
    DEFAULT_CAPTIONER_CHECKPOINT_ID,
    DEFAULT_CAPTIONER_MODEL,
    DEFAULT_GLOBAL_VL_BASE_URL,
    DEFAULT_GLOBAL_VL_CHECKPOINT_ID,
    DEFAULT_GLOBAL_VL_MODEL,
    CaptionerConfig,
    GlobalSemanticsConfig,
    LocalSemanticsConfig,
    MusicRescueConfig,
    OpenAICaptionerBackend,
    OpenAIGlobalSemanticsBackend,
    OpenAILocalSemanticsBackend,
    OpenAIMusicRescueBackend,
    build_specialized_inventory,
    run_assemble_phase,
    run_captioner_phase,
    run_global_semantics_phase,
    run_local_semantics_phase,
    run_specialized_pipeline,
    specialized_output_root,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _top_p(value: str) -> float:
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("value must be in (0, 1]")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run independently resumable H3 specialized audio semantics",
    )
    parser.add_argument("--audio-production-root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=(
            "captioner",
            "global-semantics",
            "local-semantics",
            "assemble",
            "pipeline",
        ),
        required=True,
    )
    publication_mode = parser.add_mutually_exclusive_group()
    publication_mode.add_argument("--overwrite", action="store_true")
    publication_mode.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-video", action="store_true")
    parser.add_argument("--captioner-max-inflight", type=_positive_int, default=1)
    parser.add_argument("--global-vl-max-inflight", type=_positive_int, default=4)
    parser.add_argument("--local-instruct-max-inflight", type=_positive_int, default=1)
    parser.add_argument("--timeout-seconds", type=_positive_float, default=600.0)
    parser.add_argument("--captioner-temperature", type=_positive_float, default=0.6)
    parser.add_argument("--captioner-top-p", type=_top_p, default=0.95)
    parser.add_argument("--captioner-top-k", type=_positive_int, default=20)
    parser.add_argument("--captioner-max-tokens", type=_positive_int, default=16384)
    parser.add_argument("--global-vl-max-tokens", type=_positive_int, default=2048)
    parser.add_argument("--local-instruct-max-tokens", type=_positive_int, default=2048)
    return parser


def _environment(name: str, *, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None or not value.strip():
        raise ValueError(f"missing specialized runtime input: {name}")
    return value.strip()


def _media_resolver(prefix: str) -> MediaURLResolver:
    mode = _environment(f"{prefix}_MEDIA_MODE", default="file")
    base_url = os.environ.get(f"{prefix}_MEDIA_BASE_URL")
    return MediaURLResolver(
        mode=mode,  # type: ignore[arg-type]
        media_root=Path(_environment(f"{prefix}_MEDIA_ROOT")),
        media_base_url=base_url if mode == "http" else None,
    )


def _captioner_config(arguments: argparse.Namespace) -> CaptionerConfig:
    return CaptionerConfig(
        base_url=_environment("QWEN3_OMNI_CAPTIONER_BASE_URL"),
        api_key=_environment("QWEN3_OMNI_CAPTIONER_API_KEY", default="EMPTY"),
        served_model_name=_environment(
            "QWEN3_OMNI_CAPTIONER_MODEL",
            default=DEFAULT_CAPTIONER_MODEL,
        ),
        checkpoint_id=_environment(
            "QWEN3_OMNI_CAPTIONER_CHECKPOINT_ID",
            default=DEFAULT_CAPTIONER_CHECKPOINT_ID,
        ),
        media_resolver=_media_resolver("QWEN3_OMNI_CAPTIONER"),
        timeout_seconds=arguments.timeout_seconds,
        temperature=arguments.captioner_temperature,
        top_p=arguments.captioner_top_p,
        top_k=arguments.captioner_top_k,
        max_tokens=arguments.captioner_max_tokens,
    )


def _global_config(arguments: argparse.Namespace) -> GlobalSemanticsConfig:
    return GlobalSemanticsConfig(
        base_url=_environment(
            "QWEN3_VL_AUDIO_SEMANTICS_BASE_URL",
            default=DEFAULT_GLOBAL_VL_BASE_URL,
        ),
        api_key=_environment("QWEN3_VL_AUDIO_SEMANTICS_API_KEY", default="EMPTY"),
        served_model_name=_environment(
            "QWEN3_VL_AUDIO_SEMANTICS_MODEL",
            default=DEFAULT_GLOBAL_VL_MODEL,
        ),
        checkpoint_id=_environment(
            "QWEN3_VL_AUDIO_SEMANTICS_CHECKPOINT_ID",
            default=DEFAULT_GLOBAL_VL_CHECKPOINT_ID,
        ),
        timeout_seconds=arguments.timeout_seconds,
        max_tokens=arguments.global_vl_max_tokens,
    )


def _local_config(arguments: argparse.Namespace) -> LocalSemanticsConfig:
    return LocalSemanticsConfig(
        base_url=_environment("QWEN3_OMNI_LOCAL_BASE_URL"),
        api_key=_environment("QWEN3_OMNI_LOCAL_API_KEY", default="EMPTY"),
        served_model_name=_environment(
            "QWEN3_OMNI_LOCAL_MODEL",
            default=DEFAULT_QWEN3_OMNI_MODEL,
        ),
        checkpoint_id=_environment(
            "QWEN3_OMNI_LOCAL_CHECKPOINT_ID",
            default=DEFAULT_QWEN3_OMNI_CHECKPOINT_ID,
        ),
        media_resolver=_media_resolver("QWEN3_OMNI_LOCAL"),
        include_video=arguments.include_video,
        timeout_seconds=arguments.timeout_seconds,
        max_tokens=arguments.local_instruct_max_tokens,
    )


def _music_config(arguments: argparse.Namespace) -> MusicRescueConfig:
    return MusicRescueConfig(
        base_url=_environment("QWEN3_OMNI_LOCAL_BASE_URL"),
        api_key=_environment("QWEN3_OMNI_LOCAL_API_KEY", default="EMPTY"),
        served_model_name=_environment(
            "QWEN3_OMNI_LOCAL_MODEL",
            default=DEFAULT_QWEN3_OMNI_MODEL,
        ),
        checkpoint_id=_environment(
            "QWEN3_OMNI_LOCAL_CHECKPOINT_ID",
            default=DEFAULT_QWEN3_OMNI_CHECKPOINT_ID,
        ),
        media_resolver=_media_resolver("QWEN3_OMNI_LOCAL"),
        timeout_seconds=arguments.timeout_seconds,
        max_tokens=arguments.local_instruct_max_tokens,
    )


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    if arguments.retry_failed and arguments.phase not in {
        "global-semantics",
        "local-semantics",
    }:
        raise ValueError(
            "--retry-failed is only supported for global-semantics or "
            "local-semantics"
        )
    production_root = arguments.audio_production_root.expanduser().resolve(strict=True)
    inventory = build_specialized_inventory(audio_production_root=production_root)
    output_root = specialized_output_root(production_root)
    result: dict[str, object] = {
        "phase": arguments.phase,
        "target_clip_count": inventory.target_clip_count,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "output_root": str(output_root),
        "model_calls": 0,
        "captioner_max_inflight": arguments.captioner_max_inflight,
        "global_vl_max_inflight": arguments.global_vl_max_inflight,
        "local_instruct_max_inflight": arguments.local_instruct_max_inflight,
    }
    if arguments.dry_run:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result

    previous_model_calls = 0
    if arguments.retry_failed:
        stage_name = arguments.phase.replace("-", "_")
        try:
            previous_summary = json.loads(
                (output_root / stage_name / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            previous_model_calls = int(previous_summary["model_call_count"])
        except Exception as exc:
            raise ValueError(
                f"cannot read existing {arguments.phase} retry summary"
            ) from exc

    if arguments.phase == "captioner":
        _, summary = run_captioner_phase(
            inventory=inventory,
            output_root=output_root,
            backend=OpenAICaptionerBackend(_captioner_config(arguments)),
            overwrite=arguments.overwrite,
            max_inflight=arguments.captioner_max_inflight,
        )
        result["summary"] = summary.model_dump(mode="json")
        result["model_calls"] = summary.model_call_count
    elif arguments.phase == "global-semantics":
        _, summary = run_global_semantics_phase(
            inventory=inventory,
            output_root=output_root,
            backend=OpenAIGlobalSemanticsBackend(_global_config(arguments)),
            captioner_backend=OpenAICaptionerBackend(_captioner_config(arguments)),
            music_backend=OpenAIMusicRescueBackend(_music_config(arguments)),
            overwrite=arguments.overwrite,
            retry_failed=arguments.retry_failed,
            max_inflight=arguments.global_vl_max_inflight,
            captioner_max_inflight=arguments.captioner_max_inflight,
            music_max_inflight=arguments.local_instruct_max_inflight,
        )
        result["summary"] = summary.model_dump(mode="json")
        result["model_calls"] = summary.model_call_count - previous_model_calls
    elif arguments.phase == "local-semantics":
        _, summary = run_local_semantics_phase(
            inventory=inventory,
            output_root=output_root,
            backend=OpenAILocalSemanticsBackend(_local_config(arguments)),
            overwrite=arguments.overwrite,
            retry_failed=arguments.retry_failed,
            max_inflight=arguments.local_instruct_max_inflight,
        )
        result["summary"] = summary.model_dump(mode="json")
        result["model_calls"] = summary.model_call_count - previous_model_calls
    elif arguments.phase == "assemble":
        _, summary = run_assemble_phase(
            inventory=inventory,
            output_root=output_root,
            overwrite=arguments.overwrite,
        )
        result["summary"] = summary.model_dump(mode="json")
    else:
        pipeline = run_specialized_pipeline(
            inventory=inventory,
            output_root=output_root,
            captioner_backend=OpenAICaptionerBackend(_captioner_config(arguments)),
            global_backend=OpenAIGlobalSemanticsBackend(_global_config(arguments)),
            local_backend=OpenAILocalSemanticsBackend(_local_config(arguments)),
            music_backend=OpenAIMusicRescueBackend(_music_config(arguments)),
            overwrite=arguments.overwrite,
            captioner_max_inflight=arguments.captioner_max_inflight,
            global_vl_max_inflight=arguments.global_vl_max_inflight,
            local_instruct_max_inflight=arguments.local_instruct_max_inflight,
        )
        result["pipeline"] = pipeline.model_dump(mode="json")
        result["model_calls"] = (
            pipeline.captioner_summary.model_call_count
            + pipeline.global_semantics_summary.model_call_count
            + pipeline.local_semantics_summary.model_call_count
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
