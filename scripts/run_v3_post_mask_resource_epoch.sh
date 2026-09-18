#!/usr/bin/env bash
# Opt-in resource-epoch launcher. Existing production launcher is untouched.
#
# This script does not replace scripts/run_v3_post_mask_visual_cluster.sh and is
# not accepted for formal production. It exists so the resource-epoch path can
# be benchmarked independently, per the server acceptance sequence.
set -euo pipefail

REPO="${POST_MASK_REPO:-/mnt/workspace/litengjie/data/R2V_DATA_V2}"
BASE_CONFIG="${POST_MASK_BASE_CONFIG:-}"
if [[ -z "${BASE_CONFIG}" ]]; then
  echo "POST_MASK_BASE_CONFIG (or --base-config) is required" >&2
  exit 2
fi

cd "${REPO}"
export OMP_NUM_THREADS=1
export PYTHONPATH="/mnt/workspace/litengjie/data/vendor/sam3${PYTHONPATH:+:${PYTHONPATH}}"

OPTS=(
  --base-config "${BASE_CONFIG}"
  --tag "${POST_MASK_TAG:-post-mask-v1}"
  --group-size "${POST_MASK_GROUP_SIZE:-8}"
  --rank "${RANK:-0}"
  --world-size "${WORLD_SIZE:-1}"
)
if [[ -n "${POST_MASK_ENTITY_MASK_ROOT:-}" ]]; then
  OPTS+=(--entity-mask-root "${POST_MASK_ENTITY_MASK_ROOT}")
fi
if [[ -n "${POST_MASK_JOB_RUNNER:-}" ]]; then
  OPTS+=(--job-runner "${POST_MASK_JOB_RUNNER}")
else
  OPTS+=(--dry-run)
fi

exec "${REPO}/.venv/bin/python" \
  tools/run_v3_post_mask_resource_epoch.py "${OPTS[@]}" "$@"
