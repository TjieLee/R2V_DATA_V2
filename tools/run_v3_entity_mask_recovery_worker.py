"""One persistent SAM3 backend per physical recovery GPU, across whole shards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3 import pre_qwen_production as stage2
from tools import entity_mask_recovery_plan as campaign
from tools import run_v3_entity_mask_auto as production


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-root", "output-root", "base-config", "plan", "log-file"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("rank", "world-size", "local-gpu-count", "local-slot"):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--single-shard")
    args = parser.parse_args(argv)
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("invalid physical recovery rank/world size")
    if args.local_gpu_count < 1 or not 0 <= args.local_slot < args.local_gpu_count:
        raise ValueError("invalid physical recovery GPU slot")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
        raise ValueError("recovery worker CUDA_VISIBLE_DEVICES mismatch")
    args.input_root = args.input_root.expanduser().resolve(strict=True)
    args.output_root = stage2._safe_output_root(args.output_root)
    args.log_file = args.log_file.expanduser().resolve(strict=False)
    if stage2.ALLOWED_WRITABLE_ROOT.resolve(strict=False) not in args.log_file.parents:
        raise ValueError("recovery worker log must be private")
    identity = stage2.load_config_identity(args.base_config)
    # Parent validates all source hashes. Child revalidates its own source shard
    # before processing, as well as the complete plan checksum/ownership mapping.
    plan = campaign.load_plan(
        args.plan,
        args.input_root,
        args.output_root,
        identity,
        check_source_hashes=False,
    )
    if plan.plan_sha256 != args.plan_sha256:
        raise ValueError("recovery plan changed after launcher validation")
    selected = plan.shards
    if args.single_shard:
        selected = [shard for shard in selected if shard.name == args.single_shard]
        if not selected:
            raise ValueError("single shard is not in recovery plan")
    assignments = campaign.weighted_assignment(
        selected, world_size=args.world_size, local_gpu_count=args.local_gpu_count
    )
    global_slot = args.rank * args.local_gpu_count + args.local_slot
    assigned = assignments[global_slot]
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    # Static lifetime guard only. Busy shards fail; they are never stolen/skipped.
    lock = (
        args.output_root
        / "_internal"
        / "recovery-physical"
        / f"{plan.plan_sha256}-{args.world_size}x{args.local_gpu_count}-{global_slot}.lock"
    )
    with stage2.shard_lock(lock), args.log_file.open("a", encoding="utf-8") as log:
        event_lock = threading.Lock()
        stdout = sys.stdout
        progress: dict[str, object] = {"active_shard": None, "completed_shards": 0}

        def emit(event: str, **fields: object) -> None:
            payload = {
                "event": event,
                "plan_sha256": plan.plan_sha256,
                "physical_global_slot": global_slot,
                "recovery_rank": args.rank,
                "gpu": args.gpu,
                **fields,
            }
            line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            with event_lock:
                print(line, file=stdout, flush=True)
                print(line, file=log, flush=True)

        stopped = threading.Event()

        def heartbeat() -> None:
            while not stopped.wait(30):
                emit("physical_worker_heartbeat", **progress)

        thread = threading.Thread(target=heartbeat, daemon=True)
        backend = None
        retryable = False
        thread.start()
        emit(
            "physical_worker_started",
            assigned_shards=[shard.name for shard in assigned],
        )
        try:
            for item in assigned:
                progress["active_shard"] = item.name
                emit("shard_started", **item.model_dump(mode="json"))
                path = args.input_root / "parts" / item.name
                shard = stage2.load_annotation_shard(path)
                if (
                    shard.sha256 != item.input_sha256
                    or len(shard.rows) != item.row_count
                ):
                    raise ValueError("recovery shard source identity changed")
                with stage2.shard_lock(
                    args.output_root
                    / "_internal"
                    / "recovery-shards"
                    / f"{path.stem}.lock"
                ):
                    if campaign.canonical_complete(shard, args.output_root, identity):
                        result = {"skipped": True, "rows": len(shard.rows)}
                    else:
                        with redirect_stdout(log), redirect_stderr(log):
                            if backend is None:
                                production.check_stage2_candidate_judge_health(
                                    identity.config
                                )
                                backend = stage2.default_backend_factory(
                                    identity.config
                                )
                                stage2.enable_sam3_session_reuse(
                                    backend,
                                    mode=production.PRODUCTION_SESSION_REUSE_MODE,
                                )
                            result = stage2.process_shard(
                                path,
                                output_root=args.output_root,
                                config_identity=identity,
                                backend=backend,
                                acquire_lock=False,
                                execution_identity_prevalidated=True,
                                static_owner=True,
                                chunk_rows=production.PRODUCTION_CHUNK_ROWS,
                                recover_incomplete_artifacts=True,
                            )
                if result.get("retryable") is True:
                    retryable = True
                    emit("shard_retryable", name=item.name, result=result)
                    # Continue independent assigned shards; leave this shard for restart.
                    continue
                progress["completed_shards"] = int(progress["completed_shards"]) + 1
                emit("shard_completed", name=item.name, result=result)
        except BaseException as exc:
            emit(
                "physical_worker_failed",
                error=f"{type(exc).__name__}: {exc}",
                **progress,
            )
            raise
        finally:
            stopped.set()
            thread.join()
            if backend is not None:
                with redirect_stdout(log), redirect_stderr(log):
                    stage2.close_backend(backend)
        result = {
            "retryable": retryable,
            "completed_shards": progress["completed_shards"],
            "assigned_shards": len(assigned),
        }
        emit("physical_worker_completed", **result)
        return result


if __name__ == "__main__":
    if main().get("retryable"):
        raise SystemExit(2)
