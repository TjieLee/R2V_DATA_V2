#!/usr/bin/env python3
"""Run frozen RA2VA reconcile jobs with durable, rank-owned clip shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.h3.mimo25_av_reconcile import (
    MimoCaseManifest,
    build_mimo25_reference_inventory,
)
from r2v_data_v2.h3.mimo25_backend import MimoBackendConfig, MimoMediaResolver
from r2v_data_v2.h3.mimo25_single_backend import SingleCallOpenAIMimo25Backend
from r2v_data_v2.h3.mimo25_stem_shadow import (
    StemAwareOpenAIMimo25Backend,
    build_stem_reconcile_jobs,
    usable_stem_reconcile_inventory,
)
from r2v_data_v2.h3.ra2va_production import process_reconcile_shard, run_reconcile_clip
from r2v_data_v2.h3.resolved_audio_stems import (
    downstream_stem_root,
    downstream_stem_route,
    load_stem_source,
)
from r2v_data_v2.h3.sam_audio_stem_shadow import (
    stem_shadow_root,
    validate_stem_diarization_lineage,
)
from r2v_data_v2.h3.t2va_mimo_stage_server import (
    StageMimoClient,
    StageMimoPoolClient,
    build_mimo_serve_command,
    mimo_v26_server_env,
)
from r2v_data_v2.h3.t2va_production import SHARD_SIZE, shard_name, shard_order
from r2v_data_v2.h3.visual_production_source import load_visual_production_inventory
from tools.run_h3_t2va_full_production import mimo_endpoints


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("visual-production-root", "visual-runs-root", "audio-production-root",
                 "production-root", "media-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--shadow-run-id", required=True)
    parser.add_argument("--case-manifest", type=Path)
    parser.add_argument("--stem-diarization-root", type=Path)
    parser.add_argument("--stem-asr-root", type=Path)
    parser.add_argument("--shards")
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-end", type=int)
    parser.add_argument("--shard-seed", type=int, default=20260917)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--request-workers", type=int)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--mimo-model", default="mimo-v2.5")
    parser.add_argument("--mimo-call-mode", choices=("multi", "single"), default="multi")
    parser.add_argument("--mimo-gpu-groups", default=os.environ.get("MIMO_GPU_GROUPS"))
    parser.add_argument("--mimo-ports", default=os.environ.get("MIMO_PORTS"))
    parser.add_argument("--mimo-sglang", type=Path)
    parser.add_argument("--mimo-checkpoint", type=Path)
    parser.add_argument("--mimo-startup-polls", type=int, default=360)
    parser.add_argument("--mimo-poll-interval", type=float, default=5.0)
    parser.add_argument("--mimo-cleanup-grace-seconds", type=float, default=30.0)
    parser.add_argument("--media-mode", choices=("base64", "http"), default="base64")
    parser.add_argument("--media-base-url")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _case_uids(arguments, stem_inventory) -> list[str]:
    if arguments.case_manifest:
        selected = MimoCaseManifest.model_validate_json(
            arguments.case_manifest.read_text(encoding="utf-8")
        ).clip_uids
    else:
        visual = load_visual_production_inventory(
            visual_production_root=arguments.visual_production_root,
            visual_runs_root=arguments.visual_runs_root,
        )
        eligible = {clip.identity.clip_uid for clip in visual.clips}
        selected = [uid for uid in stem_inventory.clip_uids if uid in eligible]
    chosen = set(selected)
    if [uid for uid in stem_inventory.clip_uids if uid in chosen] != selected:
        raise ValueError("RA2VA cases must be an ordered subset of frozen stem inventory")
    if not selected:
        raise ValueError("RA2VA source has no selected reference-eligible clips")
    return selected


def _runtime_paths(arguments) -> tuple[Path, Path]:
    if arguments.mimo_sglang is not None and arguments.mimo_checkpoint is not None:
        return arguments.mimo_sglang, arguments.mimo_checkpoint
    if arguments.mimo_model == "mimo-v2.6-flash-rl":
        env = Path("/mnt/workspace/litengjie/data/audio_deps/mimo26-sglang-env")
        checkpoint = Path("/mnt/workspace/public/pretrained/MiMo/MiMo-V2.6-Flash-RL")
    else:
        env = Path("/mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env")
        checkpoint = Path("/mnt/workspace/public/pretrained/MiMo/MiMo-V2.5")
    return arguments.mimo_sglang or env / "bin/sglang", arguments.mimo_checkpoint or checkpoint


def main(argv: list[str] | None = None) -> dict[str, object]:
    arguments = _parser().parse_args(argv)
    if arguments.mimo_call_mode == "single" and arguments.mimo_model != "mimo-v2.6-flash-rl":
        raise ValueError("single RA2VA production requires explicit MiMo V2.6")
    gpu_ids = arguments.gpu_ids.split(",")
    endpoints = mimo_endpoints(
        arguments.mimo_model, arguments.mimo_call_mode, gpu_ids,
        arguments.mimo_gpu_groups, arguments.mimo_ports,
    )
    workers = arguments.request_workers or (2 if endpoints else 1)
    if workers < 1 or (endpoints and workers > len(endpoints)):
        raise ValueError("RA2VA request workers exceed available MiMo endpoints")
    audio_root = arguments.audio_production_root
    shadow = stem_shadow_root(audio_root, arguments.shadow_run_id)
    stems_root = downstream_stem_root(audio_root, arguments.shadow_run_id)
    route = downstream_stem_route(arguments.shadow_run_id, None)
    diarization = arguments.stem_diarization_root or shadow / "diarization_no_lrasd_v1"
    asr = arguments.stem_asr_root or shadow / "asr_no_lrasd_v1"
    stem_inventory, stem_records, _ = load_stem_source(stems_root)
    provenance, _, _ = validate_stem_diarization_lineage(
        diarization, expected_shadow_root=shadow,
    )
    if provenance.binding_evidence_mode != "none" or Path(provenance.source_stem_root).resolve() != stems_root.resolve():
        raise ValueError("RA2VA production requires current no-LR-ASD resolved stem lineage")
    selected = _case_uids(arguments, stem_inventory)
    base = build_mimo25_reference_inventory(
        visual_production_root=arguments.visual_production_root,
        visual_runs_root=arguments.visual_runs_root,
        audio_production_root=audio_root,
        case_manifest=MimoCaseManifest(clip_uids=selected),
        verify_video=False, verify_audio=False,
    )
    jobs = build_stem_reconcile_jobs(
        base_inventory=usable_stem_reconcile_inventory(base, provenance),
        stem_diarization_root=diarization, stem_asr_root=asr,
        route=route, binding_evidence_mode="none",
    )
    by_job = {job.clip_uid: job for job in jobs}
    by_stem = {record.clip_uid: record for record in stem_records}
    skip_reasons = {
        item.clip_uid: item.reason_code
        for item in [*provenance.skipped_clips, *provenance.diarization_failed_clips]
    }
    skipped = {uid: skip_reasons.get(uid, "upstream_unusable") for uid in selected if uid not in by_job}
    last_shard = (len(selected) - 1) // SHARD_SIZE
    if arguments.shards is not None:
        explicit = [int(value) for value in arguments.shards.split(",")]
        if any(value > last_shard for value in explicit):
            raise ValueError("RA2VA shard exceeds selected source inventory")
    else:
        explicit = None
    order = shard_order(
        explicit=explicit, start=arguments.shard_start,
        end=arguments.shard_end if arguments.shard_end is not None else last_shard,
        seed=arguments.shard_seed, rank=arguments.rank, world_size=arguments.world_size,
    )
    output = arguments.production_root
    order.sort(key=lambda shard_id: (output / "shards" / shard_name(shard_id) / "COMPLETE").exists())
    result: dict[str, object] = {
        "dry_run": arguments.dry_run,
        "clip_count": len(selected), "ready_source_count": len(jobs),
        "upstream_skipped_count": len(skipped), "shards": order,
        "call_mode": arguments.mimo_call_mode, "model": arguments.mimo_model,
        "endpoints": endpoints, "request_workers": workers,
        "model_called": False, "production_root": str(output),
    }
    if arguments.dry_run:
        print(json.dumps(result, sort_keys=True))
        return result

    sglang, checkpoint = _runtime_paths(arguments)
    config = MimoBackendConfig(
        media_resolver=MimoMediaResolver(
            mode=arguments.media_mode, media_root=arguments.media_root,
            media_base_url=arguments.media_base_url,
        ),
        api_key=os.environ.get("MIMO_API_KEY", "local-no-key"),
        transport="sglang", model=arguments.mimo_model,
        base_url=f"http://127.0.0.1:{endpoints[0]['port']}/v1" if endpoints else "http://127.0.0.1:8092/v1",
        max_completion_tokens=32768,
    )

    def client(base_url: str, port: int, group: str | None = None) -> StageMimoClient:
        return StageMimoClient(
            api_key=config.api_key, base_url=base_url, timeout_seconds=config.timeout_seconds,
            serve_command=build_mimo_serve_command(
                sglang, checkpoint, served_model_name=arguments.mimo_model, port=port,
            ),
            serve_env=mimo_v26_server_env(group) if group else None,
            log_root=output / "logs" / os.uname().nodename,
            startup_polls=arguments.mimo_startup_polls,
            poll_interval=arguments.mimo_poll_interval,
            cleanup_grace_seconds=arguments.mimo_cleanup_grace_seconds,
        )

    mimo_client = (
        StageMimoPoolClient([
            client(f"http://127.0.0.1:{item['port']}/v1", item["port"], item["gpu_group"])
            for item in endpoints
        ]) if endpoints else client(config.base_url, 8092)
    )
    backend_class = SingleCallOpenAIMimo25Backend if arguments.mimo_call_mode == "single" else StemAwareOpenAIMimo25Backend
    backend = backend_class(config, stem_records_by_clip=by_stem, client=mimo_client)

    def run_clip(uid: str, destination: Path) -> dict:
        result["model_called"] = True
        return run_reconcile_clip(
            job=by_job[uid], stem_record=by_stem[uid], backend=backend,
            destination=destination, route=route,
            allow_unverified=arguments.allow_unverified,
        )

    for shard_id in order:
        clip_uids = selected[shard_id * SHARD_SIZE:(shard_id + 1) * SHARD_SIZE]
        states = process_reconcile_shard(
            root=output, shard_id=shard_id, clip_uids=clip_uids,
            run_clip=run_clip, skipped={uid: skipped[uid] for uid in clip_uids if uid in skipped},
            request_workers=workers, stage=mimo_client.stage,
        )
        print(json.dumps({
            "shard": shard_id, "ready": sum(s["status"] == "ready" for s in states.values()),
            "failed": sum(s["status"] == "failed" for s in states.values()),
            "skipped": sum(s["status"] == "skipped" for s in states.values()),
        }, sort_keys=True), flush=True)
    return result


if __name__ == "__main__":
    main()
