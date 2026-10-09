#!/usr/bin/env python3
"""Isolated eight-GPU/32-SAM-process SA execution smoke: real masks and NPY IO.

Reads an existing real-SA benchmark jobs.jsonl and the accepted SAM3 config.
No Qwen/Boogu, formal post-mask state, receipts or source mutation. Timings
exclude initial SAM model load and Triton warm-up; writing is included.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from r2v_data_v2.v3.config import load_config
from r2v_data_v2.v3.post_mask_epoch_sa_execution import SASamPipelineExecutor
from r2v_data_v2.v3.post_mask_epoch_sa_process import SAProcessPool
from r2v_data_v2.v3.post_mask_epoch_state import atomic_write_bytes


@dataclass(frozen=True)
class _Job:
    serial: int
    canonical_shard: str
    clip_uid: str
    frame_path: str
    frame_slot: int
    grounding_prompt: str
    job_type: str = "subject_attribute_sam_probe"
    attempt_index: int = 0

    def job_id(self) -> str:
        return f"sa-multiproc-smoke-{self.serial:06d}"


class _Runner:
    def __init__(self, destination: Path) -> None:
        self.destination = destination

    def prepare_sam_job(self, job: _Job) -> dict:
        return {
            "frame_path": Path(job.frame_path),
            "frame_slot": job.frame_slot,
            "grounding_prompt": job.grounding_prompt,
        }

    def infer_sam_job(self, job: _Job, prepared: dict, handle: object) -> dict:
        return {"masks": tuple(np.array(m, copy=True) for m in handle.segment_frame(**prepared))}

    def persist_sam_job(self, job: _Job, inferred: dict) -> dict:
        size = 0
        for mask_index, mask in enumerate(inferred["masks"]):
            stream = io.BytesIO()
            np.save(stream, mask, allow_pickle=False)
            payload = stream.getvalue()
            output = self.destination / job.job_id() / f"mask-{mask_index:03d}.npy"
            atomic_write_bytes(output, payload)
            size += len(payload)
        return {"masks": len(inferred["masks"]), "bytes": size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=4, choices=(2, 4))
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--persist-workers", type=int, default=8)
    args = parser.parse_args()

    # The smoke must never target any public dataset/production export root.
    allowed = Path("/mnt/workspace/litengjie").resolve()
    destination = args.output.resolve()
    if allowed not in destination.parents or destination.exists():
        raise ValueError("Need a fresh output directory under /mnt/workspace/litengjie")
    config = load_config(args.config)
    gpu_ids = tuple(int(v) for v in args.gpu_ids.split(","))
    if not gpu_ids or args.count < 1 or args.persist_workers < 1:
        raise ValueError("invalid worker count, GPU selection, or job count")
    jobs = []
    with args.manifest.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            jobs.append(_Job(
                serial=len(jobs),
                canonical_shard=record["shard"],
                clip_uid=record["clip_uid"],
                frame_path=record["frame_path"],
                frame_slot=int(record["frame_slot"]),
                grounding_prompt=record["grounding_prompt"],
            ))
            if len(jobs) == args.count:
                break
    if len(jobs) != args.count:
        raise RuntimeError(f"Only {len(jobs)} real frame/prompt jobs available")

    destination.mkdir(parents=True, exist_ok=False)
    pool = SAProcessPool(
        config.sam3, gpu_ids, workers_per_gpu=args.workers_per_gpu
    )
    executor = None
    try:
        print(f"Spawning {pool.slot_count} SAM processes, one per CUDA context",
              flush=True)
        pool.start()
        # Sequential warmup avoids loading 32 checkpoints at the same time.
        for slot in range(pool.slot_count):
            job = jobs[slot % len(jobs)]
            pool.handle_for_slot(slot).segment_frame(
                frame_path=Path(job.frame_path),
                frame_slot=job.frame_slot,
                grounding_prompt=job.grounding_prompt,
            )
            print(f"Warmed SAM process {slot + 1}/{pool.slot_count}", flush=True)
        executor = SASamPipelineExecutor(
            _Runner(destination / "masks"), resource=pool,
            slot_count=pool.slot_count, prepare_workers=8,
            persist_workers=args.persist_workers,
        )
        started = time.perf_counter()
        admitted = completed = inflight = mask_count = byte_count = 0
        while completed < len(jobs):
            while admitted < len(jobs) and inflight < executor.capacity():
                executor.submit(jobs[admitted])
                admitted += 1
                inflight += 1
            for item in executor.collect():
                inflight -= 1
                if item.exception is not None:
                    raise RuntimeError(
                        f"SAM smoke job {item.job.serial} failed"
                    ) from item.exception
                completed += 1
                mask_count += item.result["masks"]
                byte_count += item.result["bytes"]
        elapsed = time.perf_counter() - started
        result = {
            "jobs": len(jobs),
            "gpus": len(gpu_ids),
            "workers_per_gpu": args.workers_per_gpu,
            "model_processes": pool.slot_count,
            "elapsed_seconds": elapsed,
            "jobs_per_second": len(jobs) / elapsed,
            "mask_files": mask_count,
            "npy_bytes": byte_count,
            "diagnostics": executor.diagnostics(),
            "note": "SAM+IPC+NPY only, excluding startup; not SA clips/s",
        }
        (destination / "metrics.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
    finally:
        if executor is not None:
            executor.close()
        pool.close()


if __name__ == "__main__":
    main()
