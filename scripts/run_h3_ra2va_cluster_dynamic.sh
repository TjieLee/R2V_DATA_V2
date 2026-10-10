#!/usr/bin/env bash
# The cluster's global RANK selects one Coordinator; all nodes run this entry.
set +x
set -euo pipefail
REPO_ROOT="${R2VA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -z "${R2VA_REPO_ROOT:-}" && ! -f "$REPO_ROOT/tools/run_h3_ra2va_cluster_node.py" ]]; then
  REPO_ROOT=/mnt/workspace/litengjie/data/R2V_DATA_V2
fi
export R2VA_ROLE="${R2VA_ROLE:-auto}"
export R2VA_FULL_GROUP="${R2VA_FULL_GROUP:-1}"
if [[ "$R2VA_FULL_GROUP" == "1" ]]; then
  export R2VA_RUN_ID="${R2VA_RUN_ID:-ra2va-group-production-v1}"
  export R2VA_MAX_GROUPS="${R2VA_MAX_GROUPS:-0}"
else
  export R2VA_RUN_ID="${R2VA_RUN_ID:-ra2va-group-pilot20-v1}"
  export R2VA_MAX_GROUPS="${R2VA_MAX_GROUPS:-1}"
fi
export R2VA_OUTPUT_ROOT="${R2VA_OUTPUT_ROOT:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA}"
export R2VA_LIMIT="${R2VA_LIMIT:-20}"
export MAX_CLIP_DURATION_SECONDS="${MAX_CLIP_DURATION_SECONDS:-20}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export MIMO_GPU_GROUPS="${MIMO_GPU_GROUPS:-0,1,2,3;4,5,6,7}"
export MIMO_PORTS="${MIMO_PORTS:-8094,8095}"
export MIMO_MEM_FRACTION_STATIC="${MIMO_MEM_FRACTION_STATIC:-0.65}"
export ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-8}"
export R2V_PYTHON="${R2V_PYTHON:-$REPO_ROOT/.venv/bin/python}"
exec "$R2V_PYTHON" "$REPO_ROOT/tools/run_h3_ra2va_cluster_node.py" "$@"
