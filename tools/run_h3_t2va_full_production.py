"""One-node, one-shard-at-a-time target Audio -> T2VA/TA2VA supervisor."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from r2v_data_v2.h3 import t2va_full_production as full
from r2v_data_v2.h3 import t2va_production as production


def required_path(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"required local dependency path: {name}")
    return Path(value).expanduser().absolute()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shot-manifest", "clips-root", "source-videos-root", "media-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--production-root", type=Path, default=production.DEFAULT_ROOT)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--request-workers", type=int, default=1)
    parser.add_argument("--canonical-workers", type=int, default=16)
    parser.add_argument("--shards")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int)
    parser.add_argument("--shard-seed", type=int, default=20260917)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--base-url", default="http://127.0.0.1:8092/v1")
    parser.add_argument(
        "--mimo-sglang",
        type=Path,
        default=Path("/mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env/bin/sglang"),
    )
    parser.add_argument(
        "--mimo-checkpoint",
        type=Path,
        default=Path("/mnt/workspace/public/pretrained/MiMo/MiMo-V2.5"),
    )
    parser.add_argument("--mimo-model", default="mimo-v2.5")
    parser.add_argument("--mimo-call-mode", choices=("multi", "single"), default="multi")
    parser.add_argument("--mimo-mem-fraction-static", type=float, default=0.65)
    parser.add_argument("--mimo-startup-polls", type=int, default=360)
    parser.add_argument("--mimo-poll-interval", type=float, default=5.0)
    parser.add_argument("--mimo-cleanup-grace-seconds", type=float, default=30.0)
    parser.add_argument("--media-mode", choices=["base64", "http"], default="base64")
    parser.add_argument("--media-base-url")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    gpu_ids = args.gpu_ids.split(",")
    if (
        not gpu_ids
        or len(set(gpu_ids)) != len(gpu_ids)
        or any(not g.isdigit() for g in gpu_ids)
    ):
        raise ValueError("GPU IDs must be unique nonnegative physical indices")
    if args.request_workers < 1:
        raise ValueError("request workers must be positive")
    if args.canonical_workers < 1:
        raise ValueError("canonical workers must be positive")
    if args.mimo_startup_polls < 1:
        raise ValueError("MiMo startup polls must be positive")
    if args.mimo_poll_interval < 0 or args.mimo_cleanup_grace_seconds <= 0:
        raise ValueError("invalid MiMo lifecycle timing")
    if not 0 < args.mimo_mem_fraction_static <= 1:
        raise ValueError("MiMo mem fraction must be in (0, 1]")
    root = args.production_root.expanduser().resolve()
    for source in (args.shot_manifest, args.clips_root, args.source_videos_root):
        if source.resolve().is_relative_to(root):
            raise ValueError("production root cannot contain source media")
    root.mkdir(parents=True, exist_ok=True)
    if not os.access(root, os.W_OK):
        raise PermissionError(f"production root is not writable: {root}")
    if not args.mimo_model.strip():
        raise ValueError("MiMo model name must be non-empty")
    print("source_index_start", flush=True)
    index = production.build_source_index(args.shot_manifest, root)
    print(
        f"source_index_ready shards={len(index['shards'])} "
        f"records={index['source_record_count']}",
        flush=True,
    )
    explicit = [int(x) for x in args.shards.split(",")] if args.shards else None
    assigned = production.shard_order(
        explicit=explicit,
        start=args.shard_start,
        end=args.shard_end if args.shard_end is not None else len(index["shards"]) - 1,
        seed=args.shard_seed,
        rank=args.rank,
        world_size=args.world_size,
    )
    if any(s >= len(index["shards"]) for s in assigned):
        raise ValueError("shard outside source inventory")
    if explicit is None:
        print(f"resume_scan_start assigned={len(assigned)}", flush=True)
        order, resume_schedule = full.resume_first_assigned_shards(root, assigned)
    else:
        order = assigned
        resume_schedule = {
            "unfinished": 0,
            "fresh": len(order),
            "complete_skipped": 0,
            "explicit": True,
        }
    if args.dry_run:
        return {
            "dry_run": True,
            "shards": order,
            "gpu_ids": gpu_ids,
            "request_workers": args.request_workers,
            "canonical_workers": args.canonical_workers,
            "resume_schedule": resume_schedule,
            "mimo_model": args.mimo_model,
            "mimo_call_mode": args.mimo_call_mode,
            "model_call_count": 0,
        }
    print(
        "schedule "
        + json.dumps(
            {
                "rank": args.rank,
                "world_size": args.world_size,
                **resume_schedule,
                "scheduled": len(order),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not args.mimo_sglang.is_file() or not os.access(args.mimo_sglang, os.X_OK):
        raise ValueError(f"MiMo SGLang executable unavailable: {args.mimo_sglang}")
    if not args.mimo_checkpoint.is_dir():
        raise ValueError(f"MiMo checkpoint unavailable: {args.mimo_checkpoint}")
    from r2v_data_v2.h3.auk_speech_shadow import auk_configuration
    from r2v_data_v2.h3.mimo25_backend import MimoMediaResolver
    from r2v_data_v2.h3.sam_audio_stem_shadow import sam_audio_configuration
    from r2v_data_v2.h3.t2va_mimo_backend import (
        T2VAMimoBackend,
        T2VAMimoConfig,
        T2VASingleCallBackend,
    )
    from r2v_data_v2.h3.t2va_mimo_stage_server import (
        StageMimoClient,
        build_mimo_serve_command,
    )
    from r2v_data_v2.h3.ta2va_shadow import TA2VAProfileBackend

    sam_config = sam_audio_configuration(
        implementation_root=required_path("SAM_AUDIO_CODE_ROOT"),
        model_path=required_path("SAM_AUDIO_MODEL_PATH"),
        model_name=os.environ.get("SAM_AUDIO_MODEL_NAME"),
        t5_base_path=required_path("SAM_AUDIO_T5_BASE_PATH"),
        device="cuda:0",
        reranking_candidates=1,
    )
    auk_config = auk_configuration(
        python_path=required_path("AUK_PYTHON"),
        code_root=required_path("AUK_CODE_ROOT"),
        checkpoint=required_path("AUK_CHECKPOINT"),
        qwen_path=required_path("AUK_QWEN_PATH"),
        device="cuda:0",
    )
    config = T2VAMimoConfig(
        media_resolver=MimoMediaResolver(
            mode=args.media_mode,
            media_root=args.media_root,
            media_base_url=args.media_base_url,
        ),
        base_url=args.base_url,
        api_key=os.environ.get("MIMO_API_KEY", "local-no-key"),
        model=args.mimo_model,
        transport="sglang",
        call_mode=args.mimo_call_mode,
    )
    mimo_client = StageMimoClient(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
        serve_command=build_mimo_serve_command(
            args.mimo_sglang,
            args.mimo_checkpoint,
            served_model_name=args.mimo_model,
            mem_fraction_static=args.mimo_mem_fraction_static,
        ),
        log_root=root / "logs" / os.uname().nodename,
        startup_polls=args.mimo_startup_polls,
        poll_interval=args.mimo_poll_interval,
        cleanup_grace_seconds=args.mimo_cleanup_grace_seconds,
    )
    print(
        f"mimo_runtime model={args.mimo_model} checkpoint={args.mimo_checkpoint}",
        flush=True,
    )
    pipeline = full.FullPipeline(
        root=root,
        index=index,
        clips_root=args.clips_root.resolve(),
        source_videos_root=args.source_videos_root.resolve(),
        gpu_ids=gpu_ids,
        sam_configuration=sam_config,
        auk_configuration=auk_config,
        backend=(T2VASingleCallBackend if args.mimo_call_mode == "single" else T2VAMimoBackend)(
            config,
            client=mimo_client,
            verify_media=False,
        ),
        profiles=TA2VAProfileBackend(config, client=mimo_client),
        allow_unverified=args.allow_unverified,
        request_workers=args.request_workers,
        canonical_workers=args.canonical_workers,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
        mimo_lifecycle=mimo_client,
    )
    previous = signal.getsignal(signal.SIGTERM)

    def stop(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    try:
        with pipeline.node(order):
            return full.run_assigned_shards(root, order, pipeline)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    print(json.dumps(main(), sort_keys=True))
