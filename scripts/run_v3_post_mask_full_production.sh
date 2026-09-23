#!/usr/bin/env bash
set -eu

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="$repo/configs/v3_post_mask_resource_epoch_production.yaml"
entity_mask=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_mask
production=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference
qwen=/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct
rank="${RANK:-0}"
world_size="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29506}"
export TORCH_HOME=/mnt/workspace/public/.cache/
export OMP_NUM_THREADS=1
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

run=("$repo/.venv/bin/python" "$repo/tools/run_v3_post_mask_full_production.py"
  --base-config "$config" --entity-mask-root "$entity_mask"
  --rank "$rank" --world-size "$world_size" --group-size 8)
printf 'host=%s rank=%s world_size=%s\n' "$(hostname)" "$rank" "$world_size"
if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'production_root=%s\nentity_mask_root=%s\nqwen_model=%s\nboogu_backend=%s\n' \
    "$production" "$entity_mask" "$qwen" boogu_image_0_1_edit_turbo
  printf '%q ' "${run[@]}"; printf '\n'
  exit 0
fi

cd "$repo"
source "$repo/.venv/bin/activate"
source "$repo/server_env.sh"
# server_env.sh is allowed to alter the shell environment, but production must
# continue from this frozen worktree even if that file changes the cwd.
cd "$repo"
export TORCH_HOME=/mnt/workspace/public/.cache/
export OMP_NUM_THREADS=1
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
export PYTHONPATH="/mnt/workspace/litengjie/data/vendor/sam3${PYTHONPATH:+:$PYTHONPATH}"
# Pin after server_env.sh; stale shell values cannot choose a different model.
export POST_MASK_QWEN_MODEL_PATH="$qwen"
export POST_MASK_CPU_WORKERS="${POST_MASK_CPU_WORKERS:-32}"
export POST_MASK_FORMAL_PRODUCTION=1
mkdir -p "$production"
test -w "$production" || { echo "Output not writable: $production" >&2; exit 2; }
logs="$production/logs/$(hostname)"
mkdir -p "$logs"
stamp="$(date +%Y%m%d-%H%M%S)-$$"
"${run[@]}" "$@" 2>&1 | tee "$logs/production-$stamp.log"
exit "${PIPESTATUS[0]}"
