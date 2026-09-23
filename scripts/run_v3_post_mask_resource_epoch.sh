#!/usr/bin/env bash
# Opt-in resource-epoch launcher. Existing production launcher is untouched.
#
# This script does not replace scripts/run_v3_post_mask_visual_cluster.sh and is
# not accepted for formal production. It exists so the resource-epoch path can
# be benchmarked independently, per the server acceptance sequence.
set -eu

REPO="${POST_MASK_REPO:-/mnt/workspace/litengjie/data/R2V_DATA_V2}"

# True when the *user* CLI arguments already carry this option. "$@" includes
# the needle itself, so it has to be shifted off first; scanning it too would
# make every check unconditionally true. The Python parser stays the authority:
# this only decides whether the shell has to add a default, so a CLI value is
# never shadowed by an environment default.
_cli_has() {
  local needle="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "${needle}" || "${arg}" == "${needle}"=* ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -z "${POST_MASK_BASE_CONFIG:-}" ]] && ! _cli_has --base-config "$@"; then
  echo "POST_MASK_BASE_CONFIG (or --base-config) is required" >&2
  exit 2
fi

cd "${REPO}"
export OMP_NUM_THREADS=1
export PYTHONPATH="/mnt/workspace/litengjie/data/vendor/sam3${PYTHONPATH:+:${PYTHONPATH}}"

# Only environment values that actually exist become CLI options, so "$@"
# appended last can override any of them. The launcher re-exports its resolved
# values for an external job runner, so both sides share one set of roots.
OPTS=()
if [[ -n "${POST_MASK_BASE_CONFIG:-}" ]]; then
  OPTS+=(--base-config "${POST_MASK_BASE_CONFIG}")
fi
if [[ -n "${POST_MASK_TAG:-}" ]]; then
  OPTS+=(--tag "${POST_MASK_TAG}")
fi
if [[ -n "${POST_MASK_ENTITY_MASK_ROOT:-}" ]]; then
  OPTS+=(--entity-mask-root "${POST_MASK_ENTITY_MASK_ROOT}")
fi
if [[ -n "${POST_MASK_ROOT:-}" ]]; then
  OPTS+=(--post-mask-root "${POST_MASK_ROOT}")
fi
if [[ -n "${POST_MASK_GROUP_SIZE:-}" ]]; then
  OPTS+=(--group-size "${POST_MASK_GROUP_SIZE}")
fi
OPTS+=(--rank "${RANK:-0}" --world-size "${WORLD_SIZE:-1}")

# A job runner is the only thing that may execute model jobs, and --dry-run
# and --job-runner are mutually exclusive in the parser. Resolution is strictly
# CLI first, environment second, plan-only default last, so the shell never
# emits both and never overrides an explicit CLI choice.
if _cli_has --dry-run "$@"; then
  :
elif _cli_has --job-runner "$@"; then
  :
elif [[ -n "${POST_MASK_JOB_RUNNER:-}" ]]; then
  OPTS+=(--job-runner "${POST_MASK_JOB_RUNNER}")
else
  OPTS+=(--dry-run)
fi

exec "${REPO}/.venv/bin/python" \
  tools/run_v3_post_mask_resource_epoch.py "${OPTS[@]}" "$@"
