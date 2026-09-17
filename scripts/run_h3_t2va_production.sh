#!/usr/bin/env bash
# Own only this invocation's endpoint and runner; preserve all durable partials.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCTION_ROOT="${PRODUCTION_ROOT:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA}"
dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then dry_run=true; shift; fi
cd "$REPO_ROOT"

if ! "$dry_run"; then
  source "$REPO_ROOT/.venv/bin/activate"
  source "$REPO_ROOT/server_env.sh"
  unset PYTHONPATH
  export R2V_PYTHON="$(command -v python)"
fi
R2V_PYTHON="${R2V_PYTHON:-$REPO_ROOT/.venv/bin/python}"
SGLANG_ENV="${SGLANG_ENV:-/mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env}"
MIMO_CHECKPOINT="${MIMO_CHECKPOINT:-/mnt/workspace/public/pretrained/MiMo/MiMo-V2.5}"
SHOT_MANIFEST="${SHOT_MANIFEST:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl}"
JEA_CLIPS_ROOT="${JEA_CLIPS_ROOT:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped}"
JEA_SOURCE_VIDEOS_ROOT="${JEA_SOURCE_VIDEOS_ROOT:-/mnt/workspace/public/dataset/jea-video/moive-183t-0808}"

serve=("$SGLANG_ENV/bin/sglang" serve
  --model-path "$MIMO_CHECKPOINT" --served-model-name mimo-v2.5
  --host 127.0.0.1 --port 8092 --trust-remote-code
  --tp 8 --dp 2 --enable-dp-attention --enable-dp-lm-head
  --mm-enable-dp-encoder --dtype bfloat16
  --attention-backend fa3 --mm-attention-backend fa3
  --context-length 131072 --mem-fraction-static 0.65
  --chunked-prefill-size 16384 --max-running-requests 8
  --reasoning-parser mimo --tool-call-parser mimo
  --constrained-json-disable-any-whitespace --enable-deterministic-inference)

run=("$R2V_PYTHON" tools/run_h3_t2va_production.py
  --shot-manifest "$SHOT_MANIFEST" --clips-root "$JEA_CLIPS_ROOT"
  --source-videos-root "$JEA_SOURCE_VIDEOS_ROOT"
  --audio-production-root "${AUDIO_PRODUCTION_ROOT:-REQUIRED}"
  --audio-shadow-run-id "${AUDIO_SHADOW_RUN_ID:-REQUIRED}"
  --production-root "$PRODUCTION_ROOT"
  --base-url http://127.0.0.1:8092/v1 --media-root "${MEDIA_ROOT:-/mnt/workspace}"
  --request-workers "${REQUEST_WORKERS:-8}")
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
: "${AUDIO_PRODUCTION_ROOT:?Set the finalized Audio cache root}"
: "${AUDIO_SHADOW_RUN_ID:?Set its resolved Audio run ID}"
mkdir -p "$PRODUCTION_ROOT"
test -w "$PRODUCTION_ROOT" || { echo "Output not writable: $PRODUCTION_ROOT" >&2; exit 1; }

# Do not adopt or kill an unrelated endpoint already listening on this port.
"$R2V_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",8092)); s.close()'
stamp="$(date +%Y%m%d-%H%M%S)-$$"
logs="$PRODUCTION_ROOT/logs/$(hostname)"
mkdir -p "$logs"
mimo_pid=""
runner_pid=""
cleanup() {
  if [[ -n "$runner_pid" ]]; then
    kill "$runner_pid" 2>/dev/null || true
    wait "$runner_pid" 2>/dev/null || true
  fi
  if [[ -n "$mimo_pid" ]]; then
    kill "$mimo_pid" 2>/dev/null || true
    wait "$mimo_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${serve[@]}" >"$logs/mimo-$stamp.log" 2>&1 &
mimo_pid=$!
ready=false
for ((i=0; i<${MIMO_STARTUP_POLLS:-360}; i++)); do
  kill -0 "$mimo_pid" 2>/dev/null || { echo "MiMo exited; see $logs/mimo-$stamp.log" >&2; exit 1; }
  if curl -fsS --max-time 5 http://127.0.0.1:8092/v1/models >/dev/null; then ready=true; break; fi
  sleep 5
done
"$ready" || { echo "MiMo readiness timeout" >&2; exit 1; }
echo "Production log: $logs/production-$stamp.log"
"${run[@]}" >"$logs/production-$stamp.log" 2>&1 &
runner_pid=$!
status=0
wait "$runner_pid" || status=$?
runner_pid=""
exit "$status"
