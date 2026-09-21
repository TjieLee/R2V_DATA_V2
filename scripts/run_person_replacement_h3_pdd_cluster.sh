#!/usr/bin/env bash
set -euo pipefail

# One command per 8xH200 node. The platform injects zero-based RANK and WORLD_SIZE.
# Every node runs this same script against the same shared input/output roots.
#
# Mapping with the production defaults:
#   group_size=4, 8 GPUs/node -> 2 independent pair shards per node
#   pair_size=2000
#   RANK=0 -> pair 0,1 -> rows [0,3999]
#   RANK=1 -> pair 2,3 -> rows [4000,7999]
#   ...

REPO="${REPO:-/mnt/workspace/litengjie/data/R2V_DATA_V2}"
cd "$REPO"

: "${RANK:?RANK must be injected by the cluster platform (zero-based node rank)}"
: "${WORLD_SIZE:?WORLD_SIZE must be injected by the cluster platform}"

if ! [[ "$RANK" =~ ^[0-9]+$ && "$WORLD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "RANK/WORLD_SIZE must be integers: RANK=$RANK WORLD_SIZE=$WORLD_SIZE" >&2
  exit 2
fi
if (( RANK >= WORLD_SIZE )); then
  echo "RANK must be smaller than WORLD_SIZE: RANK=$RANK WORLD_SIZE=$WORLD_SIZE" >&2
  exit 2
fi

PAIR_PYTHON="${PAIR_PYTHON:-/mnt/workspace/litengjie/data/person_replacement_deps/bernini-qwen-env/bin/python}"
H3_PYTHON="${H3_PYTHON:-/mnt/workspace/litengjie/data/person_replacement_deps/minimax-h3-env/bin/python}"
PROMPT_WRITER_PYTHON="${PROMPT_WRITER_PYTHON:-/mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env/bin/python}"

PROMPT_WRITER_MODEL="${PROMPT_WRITER_MODEL:-/mnt/workspace/public/pretrained/Qwen/Qwen3.5-27B}"
H3_MODEL_ROOT="${H3_MODEL_ROOT:-/mnt/workspace/public/pretrained/MiniMaxAI/MiniMax-H3}"
H3_PDD_CODE_ROOT="${H3_PDD_CODE_ROOT:-/mnt/workspace/litengjie/data/vendor/MiniMax-H3-Acc-LoRAs}"
H3_PDD_LORA="${H3_PDD_LORA:-/mnt/workspace/litengjie/data/pretrained/alibaba-pai/MiniMax-H3-Acc-LoRAs/MiniMax-H3-Ref2VA-Acc-8Step.safetensors}"

INPUT_JSONL="${PERSON_REPLACEMENT_INPUT_JSONL:-/mnt/workspace/liutao/X_human_data/collected_face_2.jsonl}"
CLIPS_ROOT="${PERSON_REPLACEMENT_CLIPS_ROOT:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped}"
OUTPUT_ROOT="${PERSON_REPLACEMENT_OUTPUT_ROOT:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/multi_person_replace}"

GPUS="${PERSON_REPLACEMENT_GPUS:-0,1,2,3,4,5,6,7}"
PAIR_SIZE="${PERSON_REPLACEMENT_PAIR_SIZE:-2000}"
GROUP_SIZE="${PERSON_REPLACEMENT_GROUP_SIZE:-4}"
ULYSSES_DEGREE="${PERSON_REPLACEMENT_ULYSSES_DEGREE:-2}"
SEED="${PERSON_REPLACEMENT_SEED:-42}"
MAX_PREPARE_ATTEMPTS="${PERSON_REPLACEMENT_MAX_PREPARE_ATTEMPTS:-2}"
MAX_GENERATE_ATTEMPTS="${PERSON_REPLACEMENT_MAX_GENERATE_ATTEMPTS:-2}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
GPU_COUNT="${#GPU_ARRAY[@]}"
if (( GPU_COUNT == 0 || GPU_COUNT % GROUP_SIZE != 0 )); then
  echo "GPU count must be divisible by GROUP_SIZE: GPUS=$GPUS GROUP_SIZE=$GROUP_SIZE" >&2
  exit 2
fi

PAIRS_PER_NODE=$(( GPU_COUNT / GROUP_SIZE ))
PAIR_START=$(( RANK * PAIRS_PER_NODE ))
FIRST_ROW=$(( PAIR_START * PAIR_SIZE ))
LAST_ROW=$(( (PAIR_START + PAIRS_PER_NODE) * PAIR_SIZE - 1 ))
TOTAL_CLUSTER_CAPACITY=$(( WORLD_SIZE * PAIRS_PER_NODE * PAIR_SIZE ))

mkdir -p "$OUTPUT_ROOT/cluster_logs"
NODE_LOG="$OUTPUT_ROOT/cluster_logs/node-rank-$(printf '%06d' "$RANK").log"

echo "=== person-replacement H3/PDD cluster node ==="
echo "rank=$RANK world_size=$WORLD_SIZE"
echo "gpus=$GPUS group_size=$GROUP_SIZE ulysses_degree=$ULYSSES_DEGREE"
echo "pairs_per_node=$PAIRS_PER_NODE pair_start=$PAIR_START pair_size=$PAIR_SIZE"
echo "nominal_rows=$FIRST_ROW..$LAST_ROW"
echo "cluster_nominal_capacity_rows=$TOTAL_CLUSTER_CAPACITY"
echo "input=$INPUT_JSONL"
echo "output=$OUTPUT_ROOT"
echo "log=$NODE_LOG"

EXTRA_ARGS=()
case "${PERSON_REPLACEMENT_RETRY_FAILED:-0}" in
  1|true|TRUE|yes|YES) EXTRA_ARGS+=(--retry-failed) ;;
esac

"$PAIR_PYTHON" tools/person_replacement/run_h3_pdd_node.py \
  --gpus "$GPUS" \
  --pair-start "$PAIR_START" \
  --group-size "$GROUP_SIZE" \
  --output-root "$OUTPUT_ROOT" \
  --input-jsonl "$INPUT_JSONL" \
  --clips-root "$CLIPS_ROOT" \
  --pair-size "$PAIR_SIZE" \
  --seed "$SEED" \
  --resume \
  --variant text \
  --ulysses-degree "$ULYSSES_DEGREE" \
  --prompt-writer-python "$PROMPT_WRITER_PYTHON" \
  --prompt-writer-model "$PROMPT_WRITER_MODEL" \
  --h3-python "$H3_PYTHON" \
  --h3-model-root "$H3_MODEL_ROOT" \
  --pdd-code-root "$H3_PDD_CODE_ROOT" \
  --pdd-lora "$H3_PDD_LORA" \
  --max-prepare-attempts "$MAX_PREPARE_ATTEMPTS" \
  --max-generate-attempts "$MAX_GENERATE_ATTEMPTS" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$NODE_LOG"
