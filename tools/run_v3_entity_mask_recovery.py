"""Explicit recovery of frozen Entity Mask shards on available physical GPUs.

Cluster-auto uses a shared immutable plan and physical RANK/WORLD_SIZE. The
explicit logical-worker interface remains available for legacy manual recovery.
All old production/recovery owners must be stopped before starting either mode.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from r2v_data_v2.v3 import pre_qwen_production as stage2
from r2v_data_v2.v3.pre_qwen_production import (
    ALLOWED_WRITABLE_ROOT,
    STAGE2_META_SCHEMA_VERSION,
    VISUAL_ALGORITHM_FREEZE,
    ConfigIdentity,
    _safe_output_root,
    _sha256_file,
    _validate_live_config_identity,
    enumerate_annotation_shards,
    load_config_identity,
    validate_execution_identity,
    validate_sam3_session_reuse_identity,
)
from tools import entity_mask_recovery_plan as campaign
from tools import run_v3_entity_mask_auto as production
from tools.entity_mask_canonical_audit import validate_fast_materialized_row
from tools.run_v3_pre_qwen_auto import parse_gpus

# This supervisor recovers the existing 5 x 8 run, never creates a new topology.
WORLD_SIZE = 5
LOCAL_GPU_COUNT = 8
GLOBAL_WORKER_COUNT = WORLD_SIZE * LOCAL_GPU_COUNT


def recovery_inventory(
    *,
    input_root: Path,
    output_root: Path,
    identity: ConfigIdentity,
) -> dict[str, object]:
    root = _safe_output_root(output_root)
    _validate_live_config_identity(identity)
    production.validate_entity_mask_production_config(identity.config)
    paths = enumerate_annotation_shards(input_root)
    topology = production.expected_static_topology(
        input_root=input_root,
        shard_paths=paths,
        world_size=WORLD_SIZE,
        local_gpu_count=LOCAL_GPU_COUNT,
    )
    production.validate_static_topology(root, topology)
    validate_execution_identity(root, chunk_rows=production.PRODUCTION_CHUNK_ROWS)
    validate_sam3_session_reuse_identity(
        root, mode=production.PRODUCTION_SESSION_REUSE_MODE
    )
    # Read existing shard lineage, not the millions of per-clip workspaces.
    # Source hashes are streamed; no JSONL/artifact file is written by inventory.
    for path in paths:
        meta_path = root / "parts" / f"{path.stem}.meta.json"
        if not meta_path.is_file():
            if any(
                candidate.exists()
                for candidate in (
                    root / "parts" / path.name,
                    root / "_internal" / path.stem / "chunks",
                    root / "artifacts" / path.stem,
                )
            ):
                raise ValueError(f"existing shard missing metadata: {path.name}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, expected in {
            "schema_version": STAGE2_META_SCHEMA_VERSION,
            "visual_algorithm_freeze": VISUAL_ALGORITHM_FREEZE,
            "input_annotation_shard": str(path),
            "input_annotation_shard_sha256": _sha256_file(path),
            "base_config_path": str(identity.path),
            "base_config_sha256": identity.sha256,
            "base_config_fingerprint": identity.fingerprint,
        }.items():
            if meta.get(key) != expected:
                raise ValueError(
                    f"recovery shard metadata mismatch: {path.name}: {key}"
                )
    workers = []
    for worker_id in range(GLOBAL_WORKER_COUNT):
        start, end = production.balanced_shard_bounds(
            len(paths), GLOBAL_WORKER_COUNT, worker_id
        )
        unfinished = [
            path.name
            for path in paths[start:end]
            if not (root / "parts" / path.name).is_file()
        ]
        workers.append(
            {
                "global_worker_id": worker_id,
                "rank": worker_id // LOCAL_GPU_COUNT,
                "local_slot": worker_id % LOCAL_GPU_COUNT,
                "shard_start_position": start,
                "shard_end_position": end,
                "unfinished_canonical_shards": unfinished,
            }
        )
    return {
        "event": "recovery_inventory",
        "topology": topology,
        "unfinished_logical_workers": [
            worker["global_worker_id"]
            for worker in workers
            if worker["unfinished_canonical_shards"]
        ],
        "workers": workers,
    }


def selected_workers(
    value: str | None, inventory: dict[str, object]
) -> list[dict[str, object]]:
    if value is None:
        return []
    ids = [int(item.strip()) for item in value.split(",")]
    if (
        not ids
        or len(ids) != len(set(ids))
        or any(worker_id < 0 or worker_id >= GLOBAL_WORKER_COUNT for worker_id in ids)
    ):
        raise ValueError("logical-workers must be unique IDs within frozen range 0..39")
    by_id = {worker["global_worker_id"]: worker for worker in inventory["workers"]}
    return [
        by_id[worker_id]
        for worker_id in ids
        if by_id[worker_id]["unfinished_canonical_shards"]
    ]


def worker_command(
    args: argparse.Namespace,
    worker: dict[str, object],
    gpu: str,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).with_name("run_v3_entity_mask_worker.py")),
        "--input-root",
        str(args.input_root),
        "--base-config",
        str(args.base_config),
        "--output-root",
        str(args.output_root),
        "--world-size",
        str(WORLD_SIZE),
        "--local-gpu-count",
        str(LOCAL_GPU_COUNT),
        "--rank",
        str(worker["rank"]),
        "--local-slot",
        str(worker["local_slot"]),
        "--global-worker-id",
        str(worker["global_worker_id"]),
        "--gpu",
        gpu,
        "--shard-start-position",
        str(worker["shard_start_position"]),
        "--shard-end-position",
        str(worker["shard_end_position"]),
        "--recover-incomplete-artifacts",
    ]


def run_workers(
    args: argparse.Namespace,
    workers: list[dict[str, object]],
    gpus: list[str],
) -> list[int]:
    if not workers:
        return []
    pending = deque(workers)
    available = deque(gpus)
    active = {}
    processes = []
    logs = []
    codes = []
    args.log_root.mkdir(parents=True, exist_ok=True)
    with production._launcher_signal_handlers():
        try:
            while pending or active:
                while pending and available:
                    worker, gpu = pending.popleft(), available.popleft()
                    worker_id = worker["global_worker_id"]
                    log = (
                        args.log_root / f"recovery-worker-{worker_id}-gpu-{gpu}.log"
                    ).open("ab")
                    logs.append(log)
                    command = worker_command(args, worker, gpu)
                    log.write((json.dumps({"command": command}) + "\n").encode())
                    log.flush()
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                    process = subprocess.Popen(
                        command,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    processes.append(process)
                    active[process] = gpu
                for process, gpu in list(active.items()):
                    if process.poll() is not None:
                        codes.append(process.wait())
                        available.append(gpu)
                        del active[process]
                if active:
                    time.sleep(0.5)
        except production.LauncherTermination as exc:
            production.terminate_worker_processes(processes)
            raise SystemExit(128 + exc.signum) from exc
        except BaseException:
            production.terminate_worker_processes(processes)
            raise
        finally:
            for log in logs:
                log.close()
    return codes


def emit_event(**payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _canonical_failure_context(exc: Exception) -> dict[str, object]:
    """Identify only the validated row being checked, without a second audit pass."""
    traceback = exc.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code in (
            stage2._validate_materialized_row.__code__,
            validate_fast_materialized_row.__code__,
        ):
            value = frame.f_locals.get("value")
            source = frame.f_locals.get("input_row")
            if (
                isinstance(value, stage2.Stage2Row)
                and isinstance(source, dict)
                and value.source_index == source.get("source_index")
                and value.clip_uid == source.get("clip_uid")
            ):
                return {
                    "clip_uid": value.clip_uid,
                    "source_index": value.source_index,
                }
        traceback = traceback.tb_next
    return {}


def audit_canonical(
    *,
    input_root: Path,
    output_root: Path,
    identity: ConfigIdentity,
    full: bool = False,
    workers: int = 16,
) -> dict[str, object]:
    """Read-only shard-parallel audit, independent of recovery planning."""
    if workers < 1:
        raise ValueError("audit-workers must be positive")
    _, paths = campaign.frozen_topology(input_root, output_root, identity)
    known_names = {path.name for path in paths}
    # Do not silently omit a published part with no matching Stage1 shard.
    paths += [
        input_root / "parts" / part.name
        for part in sorted((output_root / "parts").glob("shard-*.jsonl"))
        if part.name not in known_names
    ]
    paths.sort(key=lambda path: path.name)
    event_lock = Lock()

    def audit_event(**payload: object) -> None:
        with event_lock:
            emit_event(**payload)

    def audit_shard(path: Path) -> dict[str, object] | None:
        part = output_root / "parts" / path.name
        if not part.exists() and not part.is_symlink():
            return None
        audit_event(event="canonical_audit_shard_started", shard=path.name)
        result = {"shard": path.name}
        try:
            if path.name not in known_names:
                raise ValueError(
                    "canonical shard has no matching completed Stage1 shard"
                )
            shard = stage2.load_annotation_shard(path)
            campaign.validate_shard_metadata(shard, output_root, identity)
            rows = stage2._validate_completed_shard(
                part,
                shard=shard,
                output_root=output_root,
                config_identity=identity,
                validate_artifacts=full,
            )
            if not full:
                for value, input_row in zip(rows, shard.rows):
                    validate_fast_materialized_row(
                        value,
                        input_row=input_row,
                        shard=shard,
                        output_root=output_root,
                        config_identity=identity,
                    )
        except Exception as exc:  # noqa: BLE001 - report each broken shard and continue auditing
            result.update(
                status="INVALID_CANONICAL",
                exception_type=type(exc).__name__,
                message=" ".join(str(exc).split())[:500],
                **_canonical_failure_context(exc),
            )
        else:
            result["status"] = "VALID_CANONICAL"
        return result

    canonical_total = valid = missing = 0
    invalid = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # map yields in source-shard order, regardless of completion order.
        for result in pool.map(audit_shard, paths):
            if result is None:
                missing += 1
                continue
            canonical_total += 1
            if result["status"] == "VALID_CANONICAL":
                valid += 1
            else:
                invalid.append(result)
            audit_event(event="canonical_audit_shard", **result)
    summary = {
        "event": "canonical_audit_completed",
        "audit_mode": "full" if full else "fast",
        "audit_workers": workers,
        "canonical_total": canonical_total,
        "valid_canonical": valid,
        "invalid_canonical": len(invalid),
        "missing_canonical": missing,
        "invalid": invalid,
    }
    emit_event(**summary)
    return summary


def _private_recovery_path(path: Path, *, args: argparse.Namespace) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if ALLOWED_WRITABLE_ROOT.resolve(strict=False) not in resolved.parents:
        raise ValueError("recovery plan/logs must stay under the private writable root")
    for protected in (args.input_root, args.output_root):
        if (
            resolved == protected
            or resolved in protected.parents
            or protected in resolved.parents
        ):
            raise ValueError(
                "recovery plan/logs must be separate from input/output roots"
            )
    return resolved


def run_cluster_recovery(
    args: argparse.Namespace, identity: ConfigIdentity
) -> dict[str, object]:
    if args.logical_workers is not None:
        raise ValueError("cluster/single-shard recovery cannot use --logical-workers")
    if not args.dry_run and not args.confirm_production_stopped:
        raise ValueError("recovery requires --confirm-production-stopped")
    rank, world_size = (
        (0, 1) if args.single_shard else production._environment_topology()
    )
    gpus = parse_gpus(args.gpus)  # Auto-detect every visible GPU by default.
    args.plan = _private_recovery_path(args.plan, args=args)
    args.log_root = _private_recovery_path(args.log_root, args=args)
    emit_event(
        event="recovery_plan_loading",
        recovery_rank=rank,
        recovery_world_size=world_size,
        plan_path=str(args.plan),
    )
    if args.dry_run and not args.plan.is_file():
        if rank != 0:
            raise ValueError("first unpublished plan dry-run must use physical rank 0")
        plan = campaign.build_plan(
            args.input_root, args.output_root, identity, emit=emit_event
        )
    else:
        plan = campaign.obtain_plan(
            args.plan,
            args.input_root,
            args.output_root,
            identity,
            rank=rank,
            timeout_seconds=args.plan_wait_seconds,
            emit=emit_event,
        )
    emit_event(
        event="recovery_plan_ready",
        plan_path=str(args.plan),
        plan_sha256=plan.plan_sha256,
        planned_shards=len(plan.shards),
        recovery_rank=rank,
        dry_run=args.dry_run,
    )
    selected = plan.shards
    if args.single_shard:
        selected = [shard for shard in selected if shard.name == args.single_shard]
        if not selected:
            raise ValueError("single shard is not in the immutable recovery campaign")
    # Never rebalance using live completion counts: every rank uses the same frozen weights.
    assignments = campaign.weighted_assignment(
        selected, world_size=world_size, local_gpu_count=len(gpus)
    )
    for slot, gpu in enumerate(gpus):
        global_slot = rank * len(gpus) + slot
        emit_event(
            event="physical_assignment",
            plan_sha256=plan.plan_sha256,
            recovery_rank=rank,
            recovery_world_size=world_size,
            recovery_local_gpu_count=len(gpus),
            local_gpu_slot=slot,
            physical_global_slot=global_slot,
            gpu=gpu,
            estimated_remaining_rows=sum(
                shard.estimated_remaining_rows for shard in assignments[global_slot]
            ),
            shards=[
                shard.model_dump(mode="json") for shard in assignments[global_slot]
            ],
        )
    report = {
        "plan_sha256": plan.plan_sha256,
        "recovery_rank": rank,
        "recovery_world_size": world_size,
        "local_gpu_count": len(gpus),
    }
    if args.dry_run:
        return report
    log_root = args.log_root / plan.plan_sha256
    log_root.mkdir(parents=True, exist_ok=True)
    worker_tool = Path(__file__).with_name("run_v3_entity_mask_recovery_worker.py")
    processes, logs = [], []
    with production._launcher_signal_handlers():
        try:
            for slot, gpu in enumerate(gpus):
                log_path = log_root / f"physical-rank-{rank}-slot-{slot}.log"
                log = log_path.open("ab")
                logs.append(log)
                command = [
                    sys.executable,
                    str(worker_tool),
                    "--input-root",
                    str(args.input_root),
                    "--output-root",
                    str(args.output_root),
                    "--base-config",
                    str(args.base_config),
                    "--plan",
                    str(args.plan),
                    "--plan-sha256",
                    plan.plan_sha256,
                    "--rank",
                    str(rank),
                    "--world-size",
                    str(world_size),
                    "--local-gpu-count",
                    str(len(gpus)),
                    "--local-slot",
                    str(slot),
                    "--gpu",
                    gpu,
                    "--log-file",
                    str(log_path),
                ]
                if args.single_shard:
                    command.extend(["--single-shard", args.single_shard])
                log.write((json.dumps({"command": command}) + "\n").encode())
                log.flush()
                processes.append(
                    subprocess.Popen(
                        command,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu),
                        # Child JSON events remain visible in the training task's stdout.
                        stdout=None,
                        stderr=log,
                    )
                )
            codes = [process.wait() for process in processes]
        except production.LauncherTermination as exc:
            production.terminate_worker_processes(processes)
            raise SystemExit(128 + exc.signum) from exc
        except BaseException:
            production.terminate_worker_processes(processes)
            raise
        finally:
            for log in logs:
                log.close()
    report.update(worker_exit_codes=codes, success=all(code == 0 for code in codes))
    emit_event(event="recovery_node_completed", **report)
    if not report["success"]:
        raise SystemExit(production.normalized_worker_exit_code(codes))
    return report


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path, default=production.PRODUCTION_INPUT_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=production.PRODUCTION_OUTPUT_ROOT
    )
    parser.add_argument(
        "--base-config", type=Path, default=production.PRODUCTION_BASE_CONFIG
    )
    parser.add_argument(
        "--log-root", type=Path, default=production.PRODUCTION_LOG_ROOT / "recovery"
    )
    parser.add_argument(
        "--logical-workers", help="Explicit comma-separated original worker IDs (0..39)"
    )
    parser.add_argument(
        "--gpus", help="Available physical GPU IDs; does not change frozen topology"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-production-stopped", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--audit-canonical",
        action="store_true",
        help="Read-only FAST metadata/path closure audit; no image content validation",
    )
    mode.add_argument(
        "--audit-canonical-full",
        action="store_true",
        help="Read-only full publication validation, including image content hashes",
    )
    mode.add_argument(
        "--cluster-auto",
        action="store_true",
        help="Recover all planned whole shards using physical RANK/WORLD_SIZE",
    )
    mode.add_argument(
        "--single-shard", help="Run just one canonical shard name from the recovery plan"
    )
    parser.add_argument(
        "--audit-workers", type=int, default=16,
        help="Parallel read-only shard audit threads (default: 16)",
    )
    parser.add_argument("--plan", type=Path, default=campaign.DEFAULT_PLAN)
    parser.add_argument("--plan-wait-seconds", type=float, default=1800)
    args = parser.parse_args(argv)
    args.input_root = args.input_root.expanduser().resolve(strict=True)
    args.output_root = _safe_output_root(args.output_root)
    identity = load_config_identity(args.base_config)
    if args.audit_canonical or args.audit_canonical_full:
        report = audit_canonical(
            input_root=args.input_root,
            output_root=args.output_root,
            identity=identity,
            full=args.audit_canonical_full,
            workers=args.audit_workers,
        )
        if report["invalid_canonical"]:
            raise SystemExit(1)
        return report
    if args.cluster_auto or args.single_shard:
        return run_cluster_recovery(args, identity)
    report = recovery_inventory(
        input_root=args.input_root, output_root=args.output_root, identity=identity
    )
    workers = selected_workers(args.logical_workers, report)
    report["selected_logical_workers"] = [
        worker["global_worker_id"] for worker in workers
    ]
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return report
    if (
        args.logical_workers is None
        or not args.gpus
        or not args.confirm_production_stopped
    ):
        raise ValueError(
            "recovery requires --logical-workers, --gpus and --confirm-production-stopped"
        )
    gpus = parse_gpus(args.gpus)
    if any(not gpu.isdecimal() for gpu in gpus):
        raise ValueError("recovery --gpus must contain physical numeric GPU IDs")
    args.log_root = args.log_root.expanduser().resolve(strict=False)
    if ALLOWED_WRITABLE_ROOT.resolve(strict=False) not in args.log_root.parents:
        raise ValueError("recovery logs must stay under the private writable root")
    for protected in (args.input_root, args.output_root):
        if (
            args.log_root == protected
            or args.log_root in protected.parents
            or protected in args.log_root.parents
        ):
            raise ValueError("recovery logs must be separate from input/output roots")
    # Preserve the existing targeted candidate-judge policy, including health check.
    if workers:
        production.check_stage2_candidate_judge_health(identity.config)
    codes = run_workers(args, workers, gpus)
    report["worker_exit_codes"] = codes
    report["success"] = all(code == 0 for code in codes)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    if not report["success"]:
        raise SystemExit(production.normalized_worker_exit_code(codes))
    return report


if __name__ == "__main__":
    main()
