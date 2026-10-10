#!/usr/bin/env bash
# Run on Node A. Native JSON owns GPUs, TP4 groups, model paths and media URLs.
set +x
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash scripts/run_h3_ra2va_group_production.sh --run-id ID --limit N --host NODE_A_ADDRESS' \
    '  --node local NODE_A.json WORKERS [--node SSH_HOST NODE_B.json WORKERS ...]' \
    '  [--port 8780] [--max-groups 1] [--output-root ROOT] [--source-root ROOT]' \
    '  [--repo-root SHARED_CHECKOUT] [--dry-run]' \
    'Set R2V_GROUP_COORDINATOR_TOKEN in the environment (never a CLI argument).' \
    'Run again with identical run-id/settings on the same Node A to resume.' \
    'JSON media_base_url must point to an explicitly provisioned per-node service;' \
    'this wrapper never starts, adopts or stops media servers.' \
    'R2V_PYTHON defaults to SHARED_CHECKOUT/.venv/bin/python on every node.'
}
fail() { printf '%s\n' "$1" >&2; exit 2; }

runtime_environment() {
  cd "$REPO_ROOT"
  export HOSTNAME="$(hostname)"
  export SAM_AUDIO_RUNTIME_PYTHONPATH="$SAM_DEPS_ROOT/sam-audio-src:$SAM_DEPS_ROOT/perception-models-src:$SAM_DEPS_ROOT/dacvae-src:$SAM_DEPS_ROOT/sam-audio-pydeps"
  export PYTHONPATH="$REPO_ROOT:$SAM_AUDIO_RUNTIME_PYTHONPATH"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  export NO_PROXY="127.0.0.1,localhost,::1,$coordinator_host"
  export no_proxy="$NO_PROXY" R2VA_RUN_ROOT R2V_PYTHON
  unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy CUDA_VISIBLE_DEVICES
}

wait_owned() {
  local pid=$1 deadline=$((SECONDS + CLEANUP_GRACE_SECONDS))
  while kill -0 "$pid" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      printf 'Owned process %s did not stop; inspect that node before resuming.\n' "$pid" >&2
      kill -KILL "$pid" 2>/dev/null || true
      break
    fi
    sleep "$LAUNCHER_POLL_SECONDS"
  done
  wait "$pid" 2>/dev/null || true
}

# Private SSH entry: token and lifetime input arrive on stdin, not in argv.
if [[ "${1:-}" == --_remote-worker ]]; then
  shift
  (( $# >= 7 )) || fail 'Incomplete remote Worker invocation'
  REPO_ROOT=$1 R2V_PYTHON=$2 SAM_DEPS_ROOT=$3 R2VA_RUN_ROOT=$4
  coordinator_host=$5 CLEANUP_GRACE_SECONDS=$6
  LAUNCHER_POLL_SECONDS=0.2
  shift 6
  IFS= read -r R2V_GROUP_COORDINATOR_TOKEN || fail 'Missing remote Coordinator token'
  [[ -n "$R2V_GROUP_COORDINATOR_TOKEN" ]] || fail 'Empty remote Coordinator token'
  export R2V_GROUP_COORDINATOR_TOKEN
  runtime_environment
  remote_pid='' input_pid=''
  remote_cleanup() {
    if [[ -n "$input_pid" ]]; then
      kill "$input_pid" 2>/dev/null || true
      wait "$input_pid" 2>/dev/null || true
    fi
    if [[ -n "$remote_pid" ]]; then
      kill "$remote_pid" 2>/dev/null || true
      wait_owned "$remote_pid"
    fi
  }
  trap remote_cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP
  "$R2V_PYTHON" "$@" </dev/null &
  remote_pid=$!
  exec 3<&0
  (
    while IFS= read -r ignored <&3; do :; done
    kill "$remote_pid" 2>/dev/null || true
  ) &
  input_pid=$!
  exec 3<&-
  status=0
  wait "$remote_pid" || status=$?
  remote_pid=''
  exit "$status"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAM_DEPS_ROOT="${SAM_DEPS_ROOT:-/mnt/workspace/litengjie/data/audio_deps}"
output_root=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/R2VA
source_root=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference/post_mask
run_id='' limit='' coordinator_host='' port=8780 max_groups=1 dry_run=false
nodes=() configurations=() worker_counts=()
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --help|-h) usage; exit 0 ;;
    --node)
      (( $# >= 4 )) || fail '--node requires HOST CONFIG WORKERS'
      nodes+=("$2"); configurations+=("$3"); worker_counts+=("$4"); shift 4 ;;
    --run-id|--limit|--host|--port|--max-groups|--output-root|--source-root|--repo-root)
      (( $# >= 2 )) || fail "Missing value for $1"
      case "$1" in
        --run-id) run_id=$2 ;; --limit) limit=$2 ;; --host) coordinator_host=$2 ;;
        --port) port=$2 ;; --max-groups) max_groups=$2 ;;
        --output-root) output_root=$2 ;; --source-root) source_root=$2 ;;
        --repo-root) REPO_ROOT=$2 ;;
      esac
      shift 2 ;;
    *) fail "Unknown option: $1" ;;
  esac
done
[[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] || fail 'An explicit safe --run-id is required'
[[ "$limit" =~ ^[1-9][0-9]*$ ]] || fail 'An explicit positive pilot --limit is required'
[[ "$max_groups" =~ ^[1-9][0-9]*$ ]] || fail '--max-groups must be positive'
[[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || fail 'Invalid Coordinator port'
[[ "$coordinator_host" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*$ && "$coordinator_host" != 0.0.0.0 ]] || fail '--host must name a concrete Node A address'
[[ "$REPO_ROOT" == /* && "$output_root" == /* && "$source_root" == /* ]] || fail 'Checkout/source/output roots must be absolute'
(( ${#nodes[@]} > 0 )) || fail 'At least --node local CONFIG WORKERS is required'
local_count=0
for ((i=0; i<${#nodes[@]}; i++)); do
  [[ "${nodes[i]}" =~ ^[A-Za-z0-9_][A-Za-z0-9_.@-]*$ ]] || fail 'Invalid SSH host'
  [[ -n "${configurations[i]}" && "${worker_counts[i]}" =~ ^[1-9][0-9]*$ ]] || fail 'Each node needs CONFIG and positive WORKERS'
  for ((j=0; j<i; j++)); do
    [[ "${nodes[j]}" != "${nodes[i]}" ]] || fail 'Duplicate node: one Native launcher per node'
  done
  if [[ "${nodes[i]}" == local ]]; then
    local_count=$((local_count + 1))
  elif [[ "$coordinator_host" == localhost || "$coordinator_host" == 127.* ]]; then
    fail 'Remote Workers require a reachable Node A --host, not loopback'
  fi
done
(( local_count == 1 )) || fail 'Exactly one --node local entry designates this host as Node A'
R2V_PYTHON="${R2V_PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ "$R2V_PYTHON" == /* && "$SAM_DEPS_ROOT" == /* ]] || fail 'Python and SAM dependency paths must be absolute and shared'
R2VA_RUN_ROOT="${output_root%/}/$run_id"
url="http://$coordinator_host:$port"
CLEANUP_GRACE_SECONDS="${CLEANUP_GRACE_SECONDS:-60}"
LAUNCHER_POLL_SECONDS="${LAUNCHER_POLL_SECONDS:-0.5}"
COORDINATOR_STARTUP_POLLS="${COORDINATOR_STARTUP_POLLS:-120}"
[[ "$CLEANUP_GRACE_SECONDS" =~ ^[1-9][0-9]*$ && "$COORDINATOR_STARTUP_POLLS" =~ ^[1-9][0-9]*$ ]] || fail 'Cleanup/startup bounds must be positive integers'
[[ "$LAUNCHER_POLL_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || fail 'Invalid launcher poll interval'
coordinator=("$R2V_PYTHON" tools/run_h3_ra2va_group_production.py coordinator
  --execution-mode native --source-root "$source_root" --output-root "$output_root"
  --run-id "$run_id" --limit "$limit" --max-groups "$max_groups"
  --host "$coordinator_host" --port "$port" --local-lock-path "/tmp/$run_id.coordinator.lock")

node_command() {
  worker=("$R2V_PYTHON" tools/run_h3_ra2va_group_production.py worker --native
    --coordinator-url "$url" --node-id "${nodes[$1]}" --workers "${worker_counts[$1]}"
    --configuration "${configurations[$1]}")
  command=("${worker[@]}")
  if [[ "${nodes[$1]}" != local ]]; then
    printf -v remote_command '%q ' bash "$REPO_ROOT/scripts/$NAME" --_remote-worker \
      "$REPO_ROOT" "$R2V_PYTHON" "$SAM_DEPS_ROOT" "$R2VA_RUN_ROOT" "$coordinator_host" \
      "$CLEANUP_GRACE_SECONDS" "${worker[@]:1}"
    command=(ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "${nodes[$1]}" "$remote_command")
  fi
}
NAME=run_h3_ra2va_group_production.sh
if "$dry_run"; then
  printf 'coordinator: '; printf '%q ' "${coordinator[@]}"; printf '\n'
  for ((i=0; i<${#nodes[@]}; i++)); do
    node_command "$i"
    printf 'worker[%s]: ' "${nodes[i]}"; printf '%q ' "${command[@]}"; printf '\n'
  done
  exit 0
fi
[[ -n "${R2V_GROUP_COORDINATOR_TOKEN:-}" ]] || fail 'Set R2V_GROUP_COORDINATOR_TOKEN before launch'
[[ "$R2V_GROUP_COORDINATOR_TOKEN" != *$'\n'* && "$R2V_GROUP_COORDINATOR_TOKEN" != *$'\r'* ]] || fail 'Coordinator token must be a single line'
export R2V_GROUP_COORDINATOR_TOKEN
runtime_environment
command -v curl >/dev/null || fail 'curl is required for Coordinator readiness'
for node in "${nodes[@]}"; do
  [[ "$node" == local ]] || command -v ssh >/dev/null || fail 'ssh is required for remote Workers'
done

# Refuse occupied ports, including unrelated services, before launching Workers.
"$R2V_PYTHON" -c 'import socket,sys; s=socket.socket(); s.bind((sys.argv[1], int(sys.argv[2]))); s.close()' "$coordinator_host" "$port"
coordinator_pid='' worker_pids=() feed_pids=()
cleanup() {
  local pid
  for pid in "${feed_pids[@]:-}"; do
    [[ -z "$pid" ]] || kill "$pid" 2>/dev/null || true
  done
  for ((i=0; i<${#worker_pids[@]}; i++)); do
    if [[ "${nodes[i]}" == local && -n "${worker_pids[i]}" ]]; then
      kill "${worker_pids[i]}" 2>/dev/null || true
    fi
  done
  for pid in "${worker_pids[@]:-}"; do
    [[ -z "$pid" ]] || wait_owned "$pid"
  done
  for pid in "${feed_pids[@]:-}"; do
    [[ -z "$pid" ]] || wait "$pid" 2>/dev/null || true
  done
  if [[ -n "$coordinator_pid" ]]; then
    kill "$coordinator_pid" 2>/dev/null || true
    wait_owned "$coordinator_pid"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
"${coordinator[@]}" </dev/null &
coordinator_pid=$!
ready=false
for ((poll=0; poll<COORDINATOR_STARTUP_POLLS; poll++)); do
  kill -0 "$coordinator_pid" 2>/dev/null || { printf 'Coordinator exited during startup\n' >&2; exit 1; }
  if printf 'Authorization: Bearer %s\n' "$R2V_GROUP_COORDINATOR_TOKEN" |
      curl --disable --silent --fail --max-time 1 --header @- --output /dev/null "$url/v1/status" 2>/dev/null; then
    kill -0 "$coordinator_pid" 2>/dev/null || exit 1
    ready=true; break
  fi
  sleep "$LAUNCHER_POLL_SECONDS"
done
"$ready" || { printf 'Coordinator readiness timed out\n' >&2; exit 1; }
for ((i=0; i<${#nodes[@]}; i++)); do
  node_command "$i"
  if [[ "${nodes[i]}" == local ]]; then
    "${command[@]}" </dev/null &
    worker_pids+=("$!")
  else
    # Keep SSH stdin open; EOF requests the remote Native Worker's own cleanup.
    exec 3< <(printf '%s\n' "$R2V_GROUP_COORDINATOR_TOKEN"; while sleep 1; do printf '\n' || break; done)
    feed_pids+=("$!")
    "${command[@]}" <&3 &
    worker_pids+=("$!")
    exec 3<&-
  fi
done
remaining=${#worker_pids[@]}
while (( remaining )); do
  for ((i=0; i<${#worker_pids[@]}; i++)); do
    pid=${worker_pids[i]}
    [[ -n "$pid" ]] || continue
    if ! kill -0 "$pid" 2>/dev/null; then
      status=0
      wait "$pid" || status=$?
      worker_pids[i]=''
      (( status == 0 )) || { printf 'Worker on %s exited (%s)\n' "${nodes[i]}" "$status" >&2; exit "$status"; }
      remaining=$((remaining - 1))
    fi
  done
  (( remaining )) || break
  kill -0 "$coordinator_pid" 2>/dev/null || { printf 'Coordinator exited before Workers completed\n' >&2; exit 1; }
  sleep "$LAUNCHER_POLL_SECONDS"
done
