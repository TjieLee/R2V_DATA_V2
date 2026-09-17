#!/usr/bin/env bash
set -eu

cd "${POST_MASK_REPO:-/mnt/workspace/litengjie/data/R2V_DATA_V2}"
export PYTHONPATH="/mnt/workspace/litengjie/data/vendor/sam3:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export POST_MASK_GPU_GROUPS="${POST_MASK_GPU_GROUPS:-4:5,6:7}"
exec .venv/bin/python tools/run_v3_post_mask_visual.py "$@"
