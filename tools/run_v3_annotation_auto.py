#!/usr/bin/env python3
import argparse
import math
import subprocess
import sys
from pathlib import Path

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--base-urls", required=True)
parser.add_argument("--rank", type=int, default=0)
parser.add_argument("--world-size", type=int, default=1)
args = parser.parse_args()

r2v = Path("/mnt/workspace/litengjie/data/R2V_DATA_V2")
source = Path("/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl")
clips = Path("/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped")
source_videos = Path("/mnt/workspace/public/dataset/jea-video/moive-183t-0808")
output = Path("/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations")
model = "/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct"
shard_size = 10000
urls = [x.strip() for x in args.base_urls.split(",") if x.strip()]

rows = 0
with source.open("rb") as f:
    for line in f:
        if not line.endswith(b"\n"):
            break
        if line.strip():
            rows += 1

total_shards = math.ceil(rows / shard_size)

def split(start, count, parts):
    base, rem = divmod(count, parts)
    ranges = []
    cur = start
    for i in range(parts):
        n = base + (1 if i < rem else 0)
        if n > 0:
            ranges.append((cur, cur + n - 1))
            cur += n
    return ranges

node_ranges = split(0, total_shards, args.world_size)
if args.rank >= len(node_ranges):
    raise SystemExit(0)

node_start, node_end = node_ranges[args.rank]
worker_ranges = split(node_start, node_end - node_start + 1, len(urls))

workers = [
    {
        "name": f"qwen_{i}",
        "base_url": urls[i],
        "shard_start": start,
        "shard_end": end,
    }
    for i, (start, end) in enumerate(worker_ranges)
]

config = {
    "source_jsonl": str(source),
    "clips_root": str(clips),
    "source_videos_root": str(source_videos),
    "output_root": str(output),
    "shard_size": shard_size,
    "annotation": {
        "model": model,
        "api_key": "EMPTY",
        "temperature": 0.0,
        "max_tokens": 4096,
        "timeout_seconds": 3600,
        "repair_retries": 1,
        "entity_selection_mode": "default",
        "fps": 2.0,
    },
    "workers": workers,
}

config_dir = Path("/mnt/workspace/litengjie/data/entity_annotation_configs")
config_dir.mkdir(parents=True, exist_ok=True)
config_path = config_dir / f"rank-{args.rank}.yaml"
config_path.write_text(
    yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
    encoding="utf-8",
)

log_dir = Path("/mnt/workspace/litengjie/data/entity_annotation_logs") / f"rank-{args.rank}"
log_dir.mkdir(parents=True, exist_ok=True)

print(f"rows={rows} shards={total_shards} rank={args.rank}/{args.world_size}")
for worker in workers:
    print(f"{worker['name']}: shard {worker['shard_start']}-{worker['shard_end']} -> {worker['base_url']}")

processes = []
logs = []
for worker in workers:
    log = (log_dir / f"{worker['name']}.log").open("ab")
    logs.append(log)
    processes.append(
        subprocess.Popen(
            [
                sys.executable,
                str(r2v / "tools/run_v3_annotation_batch.py"),
                "--config",
                str(config_path),
                "--worker",
                worker["name"],
            ],
            cwd=r2v,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    )

code = max((p.wait() for p in processes), default=0)
for log in logs:
    log.close()
raise SystemExit(code)
