"""Explicit recovery of frozen Entity Mask logical workers on available GPUs.

Normal production owners must be stopped before using this tool. Different
recovery nodes must receive disjoint explicit --logical-workers lists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

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
from tools import run_v3_entity_mask_auto as production
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
    args = parser.parse_args(argv)
    args.input_root = args.input_root.expanduser().resolve(strict=True)
    args.output_root = _safe_output_root(args.output_root)
    identity = load_config_identity(args.base_config)
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
