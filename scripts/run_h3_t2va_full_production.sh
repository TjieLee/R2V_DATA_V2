#!/usr/bin/env bash
# One endpoint for the entire full-production supervisor, including all shards.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCTION_ROOT="${PRODUCTION_ROOT:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA}"
dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
cd "$REPO_ROOT"

if ! "$dry_run"; then
  source "$REPO_ROOT/.venv/bin/activate"
  source "$REPO_ROOT/server_env.sh"

  SAM_DEPS_ROOT="${SAM_DEPS_ROOT:-/mnt/workspace/litengjie/data/audio_deps}"
  export SAM_AUDIO_CODE_ROOT="${SAM_AUDIO_CODE_ROOT:-$SAM_DEPS_ROOT/sam-audio-src}"
  PERCEPTION_MODELS_ROOT="${PERCEPTION_MODELS_ROOT:-$SAM_DEPS_ROOT/perception-models-src}"
  DACVAE_ROOT="${DACVAE_ROOT:-$SAM_DEPS_ROOT/dacvae-src}"
  SAM_AUDIO_PYDEPS="${SAM_AUDIO_PYDEPS:-$SAM_DEPS_ROOT/sam-audio-pydeps}"
  export SAM_AUDIO_RUNTIME_PYTHONPATH="${SAM_AUDIO_RUNTIME_PYTHONPATH:-$SAM_AUDIO_CODE_ROOT:$PERCEPTION_MODELS_ROOT:$DACVAE_ROOT:$SAM_AUDIO_PYDEPS}"
  export SAM_AUDIO_MODEL_PATH="${SAM_AUDIO_MODEL_PATH:-/mnt/workspace/public/pretrained/Facebook/sam-audio-large-tv}"
  export SAM_AUDIO_MODEL_NAME="${SAM_AUDIO_MODEL_NAME:-facebook/sam-audio-large-tv}"
  export SAM_AUDIO_T5_BASE_PATH="${SAM_AUDIO_T5_BASE_PATH:-/mnt/workspace/public/pretrained/google/t5-base}"

  AUK_ROOT="${AUK_ROOT:-/mnt/workspace/litengjie/data/audio_deps/auk}"
  export AUK_PYTHON="${AUK_PYTHON:-$AUK_ROOT/auk-venv/bin/python}"
  export AUK_CODE_ROOT="${AUK_CODE_ROOT:-$AUK_ROOT/AuK-src}"
  export AUK_CHECKPOINT="${AUK_CHECKPOINT:-$AUK_ROOT/AuK/auk_base.safetensors}"
  export AUK_QWEN_PATH="${AUK_QWEN_PATH:-$AUK_ROOT/Qwen2.5-Omni-3B}"

  export DIARIZEN_PYTHON="${DIARIZEN_PYTHON:-/mnt/workspace/litengjie/data/audio_deps/diarizen-venv/bin/python}"
  export DIARIZEN_CODE_ROOT="${DIARIZEN_CODE_ROOT:-/mnt/workspace/litengjie/data/audio_deps/DiariZen}"
  export DIARIZEN_MODEL_PATH="${DIARIZEN_MODEL_PATH:-/mnt/workspace/litengjie/data/audio_deps/diarizen-model-cache}"
  export DIARIZEN_MODEL_IDENTIFIER="${DIARIZEN_MODEL_IDENTIFIER:-BUT-FIT/diarizen-wavlm-large-s80-md-v2}"
  export DIARIZEN_DEVICE="${DIARIZEN_DEVICE:-cuda:0}"
  export DIARIZEN_TIMEOUT_SECONDS="${DIARIZEN_TIMEOUT_SECONDS:-900}"

  export QWEN3_ASR_ENV="${QWEN3_ASR_ENV:-/mnt/workspace/litengjie/data/audio_deps/qwen3-asr-venv}"
  export QWEN3_ASR_MODEL_PATH="${QWEN3_ASR_MODEL_PATH:-/mnt/workspace/public/pretrained/Qwen/Qwen3-ASR-1.7B}"
  export QWEN3_ASR_DEVICE="${QWEN3_ASR_DEVICE:-cuda:0}"
  export QWEN3_ASR_DTYPE="${QWEN3_ASR_DTYPE:-bfloat16}"
  export QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE="${QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE:-1}"
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}${no_proxy:+$no_proxy,}localhost,127.0.0.1,::1"
  export no_proxy="$NO_PROXY"
  unset PYTHONPATH
  export R2V_PYTHON="$(command -v python)"
fi
R2V_PYTHON="${R2V_PYTHON:-$REPO_ROOT/.venv/bin/python}"
SGLANG_ENV="${SGLANG_ENV:-/mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env}"
MIMO_CHECKPOINT="${MIMO_CHECKPOINT:-/mnt/workspace/public/pretrained/MiMo/MiMo-V2.5}"
SHOT_MANIFEST="${SHOT_MANIFEST:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl}"
JEA_CLIPS_ROOT="${JEA_CLIPS_ROOT:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped}"
JEA_SOURCE_VIDEOS_ROOT="${JEA_SOURCE_VIDEOS_ROOT:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808}"
GPU_IDS="${GPU_IDS-0,1,2,3,4,5,6,7}"
if [[ ! "$GPU_IDS" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
  echo "GPU_IDS must contain 1-8 unique nonnegative integer IDs" >&2; exit 2
fi
IFS=, read -r -a gpu_ids <<< "$GPU_IDS"
if (( ${#gpu_ids[@]} > 8 )); then
  echo "GPU_IDS must contain at most 8 IDs" >&2; exit 2
fi
seen=,
for gpu in "${gpu_ids[@]}"; do
  if [[ "$seen" == *",$gpu,"* ]]; then
    echo "GPU_IDS must be unique" >&2; exit 2
  fi
  seen+="$gpu,"
done
REQUEST_WORKERS="${REQUEST_WORKERS:-1}"
CANONICAL_WORKERS="${CANONICAL_WORKERS:-16}"
MIMO_STARTUP_POLLS="${MIMO_STARTUP_POLLS:-360}"
MIMO_POLL_INTERVAL="${MIMO_POLL_INTERVAL:-5}"
CLEANUP_GRACE_SECONDS="${CLEANUP_GRACE_SECONDS:-30}"
for setting in REQUEST_WORKERS CANONICAL_WORKERS MIMO_STARTUP_POLLS CLEANUP_GRACE_SECONDS; do
  if [[ ! "${!setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "$setting must be a positive integer" >&2; exit 2
  fi
done
if [[ ! "$MIMO_POLL_INTERVAL" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "MIMO_POLL_INTERVAL must be nonnegative seconds" >&2; exit 2
fi

serve=("$SGLANG_ENV/bin/sglang" serve
  --model-path "$MIMO_CHECKPOINT" --served-model-name mimo-v2.5
  --host 127.0.0.1 --port 8092 --trust-remote-code
  --tp 8 --dp 2 --enable-dp-attention --enable-dp-lm-head
  --mm-enable-dp-encoder --dtype bfloat16
  --attention-backend fa3 --mm-attention-backend fa3
  --context-length 131072 --mem-fraction-static 0.55
  --chunked-prefill-size 16384 --max-running-requests 8
  --reasoning-parser mimo --tool-call-parser mimo
  --constrained-json-disable-any-whitespace --enable-deterministic-inference)

run=("$R2V_PYTHON" "$REPO_ROOT/tools/run_h3_t2va_full_production.py"
  --shot-manifest "$SHOT_MANIFEST" --clips-root "$JEA_CLIPS_ROOT"
  --source-videos-root "$JEA_SOURCE_VIDEOS_ROOT"
  --production-root "$PRODUCTION_ROOT"
  --base-url http://127.0.0.1:8092/v1 --media-root "${MEDIA_ROOT:-/mnt/workspace}"
  --request-workers "$REQUEST_WORKERS" --gpu-ids "$GPU_IDS"
  --canonical-workers "$CANONICAL_WORKERS"
  --ffmpeg "${FFMPEG:-ffmpeg}")
if [[ -n "${SHARDS:-}" ]]; then
  run+=(--shards "$SHARDS")
else
  run+=(--shard-start "${SHARD_START:-0}" --shard-seed "${SHARD_SEED:-20260917}"
    --rank "${RANK:-0}" --world-size "${WORLD_SIZE:-1}")
  if [[ -n "${SHARD_END:-}" ]]; then run+=(--shard-end "$SHARD_END"); fi
fi
if [[ "${ALLOW_UNVERIFIED:-0}" == 1 ]]; then run+=(--allow-unverified); fi
run+=("$@")
if "$dry_run"; then
  printf '%q ' "${serve[@]}"; printf '\n'
  printf '%q ' "${run[@]}"; printf '\n'
  exit 0
fi

command -v setsid >/dev/null 2>&1 || {
  echo "setsid is required for full-production process isolation" >&2
  exit 2
}
test -x "$AUK_PYTHON" || {
  echo "Missing AUK_PYTHON: $AUK_PYTHON" >&2
  exit 2
}
test -d "$AUK_CODE_ROOT" || {
  echo "Missing AUK_CODE_ROOT: $AUK_CODE_ROOT" >&2
  exit 2
}
test -f "$AUK_CHECKPOINT" || {
  echo "Missing AUK_CHECKPOINT: $AUK_CHECKPOINT" >&2
  exit 2
}
test -d "$AUK_QWEN_PATH" || {
  echo "Missing AUK_QWEN_PATH: $AUK_QWEN_PATH" >&2
  exit 2
}
test -x "$DIARIZEN_PYTHON" || {
  echo "Missing DIARIZEN_PYTHON: $DIARIZEN_PYTHON" >&2
  exit 2
}
test -d "$DIARIZEN_CODE_ROOT" || {
  echo "Missing DIARIZEN_CODE_ROOT: $DIARIZEN_CODE_ROOT" >&2
  exit 2
}
test -d "$DIARIZEN_MODEL_PATH" || {
  echo "Missing DIARIZEN_MODEL_PATH: $DIARIZEN_MODEL_PATH" >&2
  exit 2
}
test -x "$QWEN3_ASR_ENV/bin/python" || {
  echo "Missing Qwen3-ASR Python: $QWEN3_ASR_ENV/bin/python" >&2
  exit 2
}
test -d "$QWEN3_ASR_MODEL_PATH" || {
  echo "Missing QWEN3_ASR_MODEL_PATH: $QWEN3_ASR_MODEL_PATH" >&2
  exit 2
}
mkdir -p "$PRODUCTION_ROOT"
test -w "$PRODUCTION_ROOT" || { echo "Output not writable: $PRODUCTION_ROOT" >&2; exit 1; }
# Never adopt an endpoint or discover processes by port/name for cleanup.
"$R2V_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8092)); s.close()'
stamp="$(date +%Y%m%d-%H%M%S)-$$"
logs="$PRODUCTION_ROOT/logs/$(hostname)"
mkdir -p "$logs"
mimo_pid=""
runner_pid=""

wait_grace() {
  local pid="$1" deadline=$((SECONDS + CLEANUP_GRACE_SECONDS))
  while kill -0 "$pid" 2>/dev/null; do
    (( SECONDS < deadline )) || return 1
    sleep 0.1
  done
}
cleanup() {
  trap '' INT TERM
  if [[ -n "$runner_pid" ]]; then
    # The supervisor handles TERM and reaps workers (including their sessions).
    kill -TERM "$runner_pid" 2>/dev/null || true
    if ! wait_grace "$runner_pid"; then
      kill -KILL -- "-$runner_pid" 2>/dev/null || true
    fi
    wait "$runner_pid" 2>/dev/null || true
    runner_pid=""
  fi
  if [[ -n "$mimo_pid" ]]; then
    kill -TERM "$mimo_pid" 2>/dev/null || true
    if ! wait_grace "$mimo_pid"; then
      kill -KILL -- "-$mimo_pid" 2>/dev/null || true
    fi
    wait "$mimo_pid" 2>/dev/null || true
    mimo_pid=""
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Isolate owned sessions without enabling shell job-control.
setsid "${serve[@]}" </dev/null >"$logs/mimo-$stamp.log" 2>&1 &
mimo_pid=$!
ready=false
for ((i=0; i<MIMO_STARTUP_POLLS; i++)); do
  kill -0 "$mimo_pid" 2>/dev/null || { echo "MiMo exited; see $logs/mimo-$stamp.log" >&2; exit 1; }
  if curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8092/v1/models >/dev/null &&
     curl --noproxy '*' -fsS --max-time 5 http://127.0.0.1:8092/model_info >/dev/null &&
     kill -0 "$mimo_pid" 2>/dev/null; then
    ready=true; break
  fi
  if (( i + 1 < MIMO_STARTUP_POLLS )); then sleep "$MIMO_POLL_INTERVAL"; fi
done
"$ready" || { echo "MiMo readiness timeout" >&2; exit 1; }
echo "Supervisor log: $logs/supervisor-$stamp.log"
setsid "${run[@]}" </dev/null >"$logs/supervisor-$stamp.log" 2>&1 &
runner_pid=$!
status=0
wait "$runner_pid" || status=$?
runner_pid=""
exit "$status"