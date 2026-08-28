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

Workers use shared-filesystem dynamic chunk claiming and nonblocking locks.
All nodes point to the same input and output roots; `RANK` only rotates scan
order and does not change durable chunk identity. Rerunning the same zero-arg
command validates and skips completed shards, resumes partial execution chunks,
and can reclaim work after a dead process releases its lock. It never reruns
Stage1 annotation or durable-complete Stage2 work.

Durable Stage2 state remains under `entity_mask/parts`, `artifacts`,
`_internal`, `locks`, and `failures`. Do not restore Python per-pixel RLE,
enable frame prefetch or request timing, or change the frozen Visual pipeline
for this production run.
