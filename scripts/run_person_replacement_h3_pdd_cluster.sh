#!/usr/bin/env bash

# One command per 8xH200 node. The platform injects zero-based RANK and WORLD_SIZE.
# Every node runs this same script against the same shared input/output roots.
#
# Mapping with the production defaults:
#   group_size=4, 8 GPUs/node -> 2 independent pair shards per node
#   pair_size=2000
#   first wave: RANK=0 -> pair 0,1; RANK=1 -> pair 2,3; ...
#   next wave advances by WORLD_SIZE * pairs_per_node:
#     RANK=0 -> pair 10,11; RANK=1 -> pair 12,13; ... when WORLD_SIZE=5.
#   Each node continues wave-by-wave until the input JSONL is exhausted.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO="${REPO:-$DEFAULT_REPO}"
cd "$REPO"
CODE_SHA="$(git rev-parse HEAD)"

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

PAIR_PYTHON="${PAIR_PYTHON:-/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python}"
H3_PYTHON="${H3_PYTHON:-/mnt/workspace/litengjie/data/person_replacement_deps/minimax-h3-env/bin/python}"
PROMPT_WRITER_PYTHON="${PROMPT_WRITER_PYTHON:-/mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env/bin/python}"

for PYTHON_BIN in "$PAIR_PYTHON" "$H3_PYTHON" "$PROMPT_WRITER_PYTHON"; do
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Required Python runtime is missing or not executable: $PYTHON_BIN" >&2
    exit 2
  fi
done

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
FIRST_PAIR_START=$(( RANK * PAIRS_PER_NODE ))
PAIR_STRIDE=$(( WORLD_SIZE * PAIRS_PER_NODE ))
TOTAL_ROWS="$(awk 'NF {n++} END {print n+0}' "$INPUT_JSONL")"
TOTAL_PAIRS=$(( (TOTAL_ROWS + PAIR_SIZE - 1) / PAIR_SIZE ))

mkdir -p "$OUTPUT_ROOT/cluster_logs"
NODE_LOG="$OUTPUT_ROOT/cluster_logs/node-rank-$(printf '%06d' "$RANK").log"

echo "=== person-replacement H3/PDD cluster node ==="
echo "repo=$REPO"
echo "code_sha=$CODE_SHA"
echo "rank=$RANK world_size=$WORLD_SIZE"
echo "gpus=$GPUS group_size=$GROUP_SIZE ulysses_degree=$ULYSSES_DEGREE"
echo "pairs_per_node=$PAIRS_PER_NODE first_pair_start=$FIRST_PAIR_START pair_stride=$PAIR_STRIDE pair_size=$PAIR_SIZE"
echo "input_rows=$TOTAL_ROWS total_pairs=$TOTAL_PAIRS"
echo "input=$INPUT_JSONL"
echo "output=$OUTPUT_ROOT"
echo "log=$NODE_LOG"

EXTRA_ARGS=()
case "${PERSON_REPLACEMENT_RETRY_FAILED:-0}" in
  1|true|TRUE|yes|YES) EXTRA_ARGS+=(--retry-failed) ;;
esac

PAIR_START="$FIRST_PAIR_START"
while (( PAIR_START < TOTAL_PAIRS )); do
  REMAINING_PAIRS=$(( TOTAL_PAIRS - PAIR_START ))
  ACTIVE_PAIRS="$PAIRS_PER_NODE"
  if (( REMAINING_PAIRS < ACTIVE_PAIRS )); then
    ACTIVE_PAIRS="$REMAINING_PAIRS"
  fi
  FIRST_ROW=$(( PAIR_START * PAIR_SIZE ))
  LAST_ROW=$(( (PAIR_START + ACTIVE_PAIRS) * PAIR_SIZE - 1 ))
  if (( LAST_ROW >= TOTAL_ROWS )); then
    LAST_ROW=$(( TOTAL_ROWS - 1 ))
  fi

  echo "=== shard wave ==="
  echo "pair_start=$PAIR_START active_pairs=$ACTIVE_PAIRS rows=$FIRST_ROW..$LAST_ROW"

  env \
    -u PYTHONPATH \
    -u PYTHONHOME \
    -u VIRTUAL_ENV \
    -u CONDA_PREFIX \
    -u CONDA_DEFAULT_ENV \
    -u PYTHONUSERBASE \
    PYTHONNOUSERSITE=1 \
    "$PAIR_PYTHON" tools/person_replacement/run_h3_pdd_node.py \
    --gpus "$GPUS" \
    --pair-start "$PAIR_START" \
    --pair-count "$ACTIVE_PAIRS" \
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

  NODE_RC=${PIPESTATUS[0]}
  if (( NODE_RC != 0 )); then
    echo "Fatal node wave failure: pair_start=$PAIR_START rc=$NODE_RC" >&2
    exit "$NODE_RC"
  fi

  PAIR_START=$(( PAIR_START + PAIR_STRIDE ))
done

echo "All assigned pair shards are settled for rank=$RANK."
exit 0
