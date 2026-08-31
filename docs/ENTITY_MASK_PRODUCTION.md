# Entity Mask Stage2 Production Runbook

This runbook covers the standalone Visual Stage2 production job only. It
consumes frozen Stage1 entity annotations and stops before foreground removal.

## Production contract

```text
input:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations

output:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_mask

private validated config:
/mnt/workspace/litengjie/data/entity_mask_configs/production.yaml

private worker logs:
/mnt/workspace/litengjie/data/entity_mask_logs
```

The output is an exact standalone-production exception. It does not make the
public dataset tree generally writable, and normal V3 `V3Config` and
`RunStorage` public-root protection remains in force.

Prepare the private config once from the server-validated Stage2 config:

```bash
mkdir -p /mnt/workspace/litengjie/data/entity_mask_configs
cp /mnt/workspace/litengjie/data/stage2_no_diagnostics.yaml \
  /mnt/workspace/litengjie/data/entity_mask_configs/production.yaml
```

The launcher fails closed if this file is missing or if its frozen production
semantics do not match diagnostics-off, progressive SAM3 rescue, and the
central Qwen candidate judge configuration.

## Execution

The external operator entry remains:

```bash
bash stage_entity_mask_run_v2.sh
```

That shell initializes the environment and invokes the same zero-argument
Python launcher shown below.

Use the same shell environment as Stage1:

```bash
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29506}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export TORCH_HOME=/mnt/workspace/public/.cache/

source /mnt/workspace/liutao/miniconda3/etc/profile.d/conda.sh
conda activate dreamidv

export CUDA_HOME=/usr/local/cuda
export CPATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/include:${CPATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}

cd /mnt/workspace/litengjie/data/R2V_DATA_V2

/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python \
  /mnt/workspace/litengjie/data/R2V_DATA_V2/tools/run_v3_entity_mask_auto.py
```

No production arguments are required. `RANK` and `WORLD_SIZE` remain
environment-driven and default to `0` and `1`. Every node auto-detects its
visible local GPUs and starts one persistent SAM3 worker per GPU. With eight
visible GPUs, all eight GPUs are reserved for SAM3; do not start local vLLM
servers on these workers.

The targeted ambiguity judge uses the centralized gateway:

```text
http://6.167.57.88:8000/v1
/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
```

Startup fails if its `/models` health check is unavailable. The launcher never
turns off Qwen anchor selection as a fallback.

## Fixed production settings and restart

```text
execution chunk rows:       100
frame prefetch workers:     0
SAM3 request timing:        disabled
SAM3 session reuse:         clip_reset_v1
debug diagnostics:          disabled
binary-mask RLE:            vectorized NumPy implementation
logical output shard rows:  10000
```

Entity Mask production no longer uses dynamic cross-node chunk claiming or
work stealing. That older `--claim-loop` scheduler remains available for other
Stage2 uses but is deprecated for this production output.

At startup, the launcher sorts completed Stage1 logical shards and computes:

```text
global_worker_count = WORLD_SIZE * visible_local_gpu_count
global_worker_id    = RANK * visible_local_gpu_count + local_gpu_slot
```

It then applies a deterministic contiguous balanced split. Every complete
10,000-row logical shard belongs to exactly one global GPU worker. Each worker
loads one persistent SAM3 backend and sequentially processes its fixed shard
range. The existing 100-row durable execution chunks, per-clip checkpoints,
resume behavior, and canonical 10,000-row compaction remain unchanged inside
each owned shard.

Every node must expose the same number of GPUs. Rank 0 initializes the existing
execution/session markers plus the static topology marker; other ranks wait up
to 120 seconds and validate the same `WORLD_SIZE`, local GPU count, and ordered
shard list. A mismatch fails closed instead of producing overlapping or missing
assignments.

Rerunning with the same topology assigns the same shard ranges, validates and
skips completed canonical shards, and resumes valid partial chunks/checkpoints.
There is intentionally no automatic shard stealing if a node is absent. It
never reruns Stage1 annotation or durable-complete Stage2 work, and it never
deletes or repairs artifacts from an earlier failed run.

Use the lightweight scheduling preflight without loading SAM3, scanning
annotation rows, or inventorying artifacts:

```bash
/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python \
  /mnt/workspace/litengjie/data/R2V_DATA_V2/tools/run_v3_entity_mask_auto.py \
  --dry-run
```

Durable Stage2 state remains under `entity_mask/parts`, `artifacts`,
`_internal`, `locks`, and `failures`. Do not restore Python per-pixel RLE,
enable frame prefetch or request timing, or change the frozen Visual pipeline
for this production run.
