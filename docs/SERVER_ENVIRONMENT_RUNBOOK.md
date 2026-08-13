# R2V V3 Server Environment Runbook

Last updated: 2026-08-13

This file is the single source of truth for the Linux server environment used by
the V3 pipeline. Update it whenever a runtime, model path, service endpoint, GPU
assignment, launch command, or isolation rule changes. Do not reconstruct these
values from chat history.

`CONFIRMED` means the value is backed by repository/server evidence. `UNVERIFIED`
means it must be checked on the server before use. Do not replace an unverified
value with a guess.

Current visual V3 development branch:

```text
feature/v3-runtime-integrity-v1
```

Current validated code baseline before this documentation update:

```text
32a9e0e17598b6bd2d7912b6fafdb08d81187285
```

Documentation-only commits may move repository HEAD. Record both the code
baseline and the docs HEAD in freeze notes.

## 1. Runtime Layers

```text
PRODUCTION
  main R2V .venv
    - V3 orchestration
    - SAM3 source through PYTHONPATH/in-process worker path
    - OpenAI-compatible Qwen client
  external service
    - vLLM serving Qwen3-VL-32B-Instruct
  separate runtime
    - Boogu Image worker

AUDIT ONLY
  main R2V .venv
    - optional embedding/audit utilities
  isolated reference-filter venv
    - SCRFD / MediaPipe pose tools
```

Do not install SAM3 into the pose venv, run the production pipeline from the
pose venv, or import the Boogu runtime into the main R2V process.

## 2. Confirmed Paths

```text
R2V repo:
  /mnt/workspace/litengjie/data/R2V_DATA_V2

R2V Python:
  /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python

SAM3 code:
  /mnt/workspace/litengjie/data/vendor/sam3

SAM3 checkpoint:
  /mnt/workspace/public/pretrained/facebook/sam3/sam3.pt

Qwen3-VL model:
  /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct

Qwen endpoint:
  http://127.0.0.1:8000/v1

Boogu code:
  /mnt/workspace/litengjie/data/vendor/Boogu-Image

Boogu Python:
  /mnt/workspace/litengjie/data/venvs/boogu-image/bin/python

Boogu model:
  /mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708

Object-Remover LoRA:
  /mnt/workspace/litengjie/data/models/Qwen-Image-Edit-2511-Object-Remover/Qwen-Image-Edit-2511-Object-Remover.safetensors
```

Writable roots:

```text
/mnt/workspace/litengjie/data/r2v_v3_configs
/mnt/workspace/litengjie/data/r2v_v3_runs
/mnt/workspace/litengjie/data/r2v_v3_exports
/mnt/workspace/litengjie/data/r2v_v3_logs
/mnt/workspace/litengjie/data/r2v_v3_audits
/mnt/workspace/litengjie/data/r2v_v3_reviews
/mnt/workspace/litengjie/data/cache
/mnt/workspace/litengjie/data/tmp
```

Read-only roots:

```text
/mnt/workspace/public/dataset
/mnt/workspace/public/pretrained
```

Never place caches, lock files, temporary outputs, or modified artifacts under
the public roots.

## 3. Preferred GPU Allocation

```text
physical GPU 0-3  Qwen3-VL vLLM tensor parallel 4
physical GPU 4    Object remover
physical GPU 5    SAM3
physical GPU 6    Boogu
physical GPU 7    spare
```

Do not globally export `CUDA_VISIBLE_DEVICES` in the normal production shell.
Use explicit stage/device configuration or a stage-local launch rule.

Important ordinal rule:

- staged removal uses the configured physical `cuda:4`; do not wrap the main
  process with `CUDA_VISIBLE_DEVICES=4`;
- an isolated SAM3-only staged invocation may expose physical GPU 5 and use
  worker-local `cuda`;
- Boogu may expose physical GPU 6 to its isolated runtime where it appears as
  local `cuda:0`.

For a full multi-stage process, inspect the active config instead of assuming a
visible-device remap.

## 4. Fresh Production Shell

Do not enable `set -e`, `set -u`, or `set -o pipefail` in an interactive server
terminal. A failed inspection command can otherwise terminate the shell.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/activate

set +e
set +u
set +o pipefail

export R2V_REPO_ROOT=/mnt/workspace/litengjie/data/R2V_DATA_V2
export MAIN_PYTHON=$R2V_REPO_ROOT/.venv/bin/python

export SAM3_CODE_ROOT=/mnt/workspace/litengjie/data/vendor/sam3
export SAM3_MODEL_PATH=/mnt/workspace/public/pretrained/facebook/sam3/sam3.pt

export QWEN_MODEL_PATH=/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
export QWEN_BASE_URL=http://127.0.0.1:8000/v1

export BOOGU_CODE_ROOT=/mnt/workspace/litengjie/data/vendor/Boogu-Image
export BOOGU_PYTHON=/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python
export BOOGU_MODEL_PATH=/mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708

export REMOVER_LORA=/mnt/workspace/litengjie/data/models/Qwen-Image-Edit-2511-Object-Remover/Qwen-Image-Edit-2511-Object-Remover.safetensors

export V3_CONFIG_ROOT=/mnt/workspace/litengjie/data/r2v_v3_configs
export V3_RUN_ROOT=/mnt/workspace/litengjie/data/r2v_v3_runs
export V3_EXPORT_ROOT=/mnt/workspace/litengjie/data/r2v_v3_exports
export V3_LOG_ROOT=/mnt/workspace/litengjie/data/r2v_v3_logs
export V3_AUDIT_ROOT=/mnt/workspace/litengjie/data/r2v_v3_audits
export V3_REVIEW_ROOT=/mnt/workspace/litengjie/data/r2v_v3_reviews

export HF_HOME=/mnt/workspace/litengjie/data/cache/huggingface
export TORCH_HOME=/mnt/workspace/litengjie/data/cache/torch
export XDG_CACHE_HOME=/mnt/workspace/litengjie/data/cache/xdg
export TMPDIR=/mnt/workspace/litengjie/data/tmp

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export PYTHONPATH="$SAM3_CODE_ROOT:$R2V_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

A machine-local `server_env.sh` may contain these exports but is not tracked.
Never put credentials in this document.

## 5. Repository and Fixed-Path Preflight

```bash
cd "$R2V_REPO_ROOT"

git fetch origin
git switch feature/v3-runtime-integrity-v1
git pull --ff-only

printf 'head=%s\nbranch=%s\npython=%s\n' \
  "$(git rev-parse HEAD)" \
  "$(git branch --show-current)" \
  "$MAIN_PYTHON"

git status --short

test -x "$MAIN_PYTHON" || echo 'ERROR: missing main Python'
test -d "$SAM3_CODE_ROOT" || echo 'ERROR: missing SAM3 source'
test -f "$SAM3_MODEL_PATH" || echo 'ERROR: missing SAM3 checkpoint'
test -d "$QWEN_MODEL_PATH" || echo 'ERROR: missing Qwen model'
test -x "$BOOGU_PYTHON" || echo 'ERROR: missing Boogu Python'
test -d "$BOOGU_CODE_ROOT" || echo 'ERROR: missing Boogu source'
test -d "$BOOGU_MODEL_PATH" || echo 'ERROR: missing Boogu model'
test -f "$REMOVER_LORA" || echo 'ERROR: missing Object-Remover LoRA'
```

Verify SAM3 import origin:

```bash
"$MAIN_PYTHON" - <<'PY'
from pathlib import Path
import sam3

package = Path(sam3.__file__).resolve()
expected = Path('/mnt/workspace/litengjie/data/vendor/sam3').resolve()
print('sam3 package:', package)
if expected not in package.parents:
    raise SystemExit(f'Unexpected SAM3 package: {package}')
print('SAM3 import PASS')
PY
```

Verify Boogu separately without inheriting the production `PYTHONPATH`:

```bash
env -u PYTHONPATH "$BOOGU_PYTHON" - <<'PY'
import sys
import torch
print('Python', sys.version)
print('torch', torch.__version__)
PY
```

## 6. Qwen3-VL / vLLM

Confirmed serving values:

```text
model:              /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
host:               127.0.0.1
port:               8000
tensor parallel:    4
max model length:   32768
max sequences:      1
GPU memory fraction: 0.90
enforce eager:      yes
```

Confirmed cold-start command:

```bash
/usr/bin/python3 /usr/local/bin/vllm serve \
  /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct \
  --served-model-name /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --media-io-kwargs '{"video":{"num_frames":-1}}' \
  --allowed-local-media-path /mnt/workspace/public/dataset \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager
```

Health check:

```bash
curl -fsS --noproxy '*' http://127.0.0.1:8000/v1/models
```

Do not copy an old 8B README example into production. Do not infer dtype,
`trust_remote_code`, or a global `CUDA_VISIBLE_DEVICES` setting that was not
explicitly deployed.

Inspect the real service before terminating anything:

```bash
ps -ef | grep '[v]llm'
ps -ef | grep '[a]pi_server'
```

Terminate only the specific verified PID with `kill -TERM`. Never use a broad
`pkill -f python`.

## 7. Current Production/Frozen Configuration Contract

The current visual V3 defaults that must not be silently changed during final
integrity validation are:

```yaml
source:
  selection_mode: fixed_selection_v1

sam3:
  anchor_search_mode: progressive_v1
  object_rescue_mode: phrase_retry_v1
  not_found_rescue_mode: entity_phrase_retry_v1
  multi_instance_rescue_mode: qwen_anchor_select_v1

coverage:
  required_visible_frames: 7

pair:
  max_candidates_per_entity: 3
  crop_padding_ratio: 0.08
  reference_prefilter_mode: conservative_v1
  background_final_guard_mode: qwen_v1

remove:
  inference_profile: object_remover_4step_v1
  num_inference_steps: 4
  device: cuda:4

reference_edit:
  scale_collapse_fallback_guard_mode: qwen_v1

reference_integrity:
  enabled: true
  mode: targeted_qwen_v1
```

Current fixed 120 selection:

```text
/mnt/workspace/litengjie/data/r2v_v3_selections/density120-fixed.json
```

Same-parent cross-pair remains disabled for the frozen visual V3 path.

## 8. Full Stage Order and Launch

The current complete stage order is:

```text
manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,export
```

Do not use old launch snippets that omit `reference_integrity`.

Set exact paths for the selected run:

```bash
CONFIG=/mnt/workspace/litengjie/data/r2v_v3_configs/<exact-config>.yaml
RUN=/mnt/workspace/litengjie/data/r2v_v3_runs/<exact-run>
LOG=/mnt/workspace/litengjie/data/r2v_v3_logs/<exact-run>.log

mkdir -p /mnt/workspace/litengjie/data/r2v_v3_logs

test -f "$CONFIG" || echo 'ERROR: missing config'
curl -fsS --noproxy '*' "$QWEN_BASE_URL/models" >/dev/null || echo 'ERROR: Qwen unhealthy'
git rev-parse HEAD
git branch --show-current
```

Launch a fresh full run:

```bash
nohup "$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,export \
  --profile \
  > "$LOG" 2>&1 &

echo $!
```

Use a fresh run root. `run.json` binds run ID, Git commit, config hash, model
identifiers, and source manifest path. Never weaken this identity check.

## 9. Bounded Stage Replays

Prefer the smallest replay that can validate a change.

SAM3 rescue A/B when annotation/frames are already prepared:

```bash
"$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages segment,rank \
  --profile
```

Reference-integrity-only replay when references and reference-edit artifacts are
already prepared:

```bash
"$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages reference_integrity \
  --profile
```

Do not rerun Qwen annotation, SAM3, remover, or Boogu merely because final
integrity code changed.

### Historical schema rule

Old runs may fail current full `ClipRecord` validation because integrity schemas
became stricter. For read-only evidence extraction from an old run, parse raw
JSON and validate only the stable sections needed by the task, for example
annotation, sampled frames, and tracked masks.

For a replay that writes current artifacts, use a fresh run root with a matching
current `run.json`. Do not delete `run.json` from a non-empty root. If old
integrity already rejected references, preserve those rejections monotonically
when preparing a strict-only replay and remove stale reference-edit entries that
no longer point to ready references. Require `ClipRecord.model_validate()` before
model calls.

## 10. Current Final-Integrity Features

The final stage now includes:

- high-confidence deterministic semantic-policy rejection before Qwen;
- one structured-output repair retry for malformed/truncated integrity JSON;
- conservative `source_bbox_fallback_v1` for artifact-only failures;
- explicit real-pixel provenance for accepted source bbox references;
- existing Qwen review for represented content and other ambiguous semantic or
  visual integrity cases.

Deterministic semantic-policy reject never invokes source bbox fallback.

The primary new counters are:

```text
semantic_policy_rejected
judge_failed
source_bbox_fallback_attempted
source_bbox_fallback_accepted
source_bbox_fallback_rejected
source_bbox_fallback_judge_failed
```

## 11. Monitoring

```bash
pgrep -af 'run_pipeline_v3.py'

nvitop
nvidia-smi

tail -n 100 "$LOG"
tail -f "$LOG"

find "$RUN/clips" -name masks.rle.json | wc -l
find "$RUN/clips" -name clip.json | wc -l
```

SAM3 may appear as memory owned by the main pipeline process instead of a
separate process name.

Profiling files live under:

```text
$RUN/profiling/events.jsonl
$RUN/profiling/summary.json
```

A missing profiling file may simply mean the run was not launched with
`--profile`; it is not alone proof of model failure.

## 12. Current Freeze Evidence Paths

SAM3 rescue A/B:

```text
run:    /mnt/workspace/litengjie/data/r2v_v3_runs/sam3-rescue120-20260813-110929
config: /mnt/workspace/litengjie/data/r2v_v3_configs/sam3-rescue120-20260813-110929.yaml
```

Latest targeted bbox/integrity run:

```text
run: /mnt/workspace/litengjie/data/r2v_v3_runs/integrity-bbox-targeted-20260813-174737
```

The exact real-run results and remaining five-case semantic check are maintained
in `V3_RUNTIME_INTEGRITY_STATE.md`.

## 13. Failure Signatures

### `existing run.json does not match the requested V3 run`

Commit, config hash, model identifiers, manifest, or run ID differs. Use a fresh
run root rather than weakening validation.

### Historical clip fails newest schema

Do not treat this as evidence that the old SAM masks or frames are corrupt.
Extract only required stable sections for read-only analysis, or construct a
fresh current replay root with consistent downstream state.

### Qwen integrity JSON EOF / schema error

Current code retries one structured-output repair. If both attempts fail, the
entity fails closed and diagnostics retain all raw responses/finish reasons.
Do not route a judge failure into bbox fallback.

### No separate SAM3 process

Expected when SAM3 runs in-process. Inspect the main pipeline process and GPU
memory before diagnosing a missing worker.

### `nohup: ignoring input`

Normal `nohup` behavior.

## 14. Audit-Only Isolation

The reference-filter/pose environment remains isolated from production:

```text
/mnt/workspace/litengjie/data/venvs/r2v-reference-filter/bin/python
```

Use `env -u PYTHONPATH` for pose/SCRFD/MediaPipe commands. Do not install SAM3
there. DINO/SigLIP/SCRFD/MediaPipe observations remain audit evidence and do not
become production thresholds without an explicit reviewed change.

Specialized historical audit procedures remain documented in the repository's
V3 audit/reference-filter documents and Git history. They are not required for
the final integrity replay.

## 15. Safety and Maintenance

Before giving or executing server commands:

1. inspect this file and the current branch/HEAD;
2. verify every fixed path with bounded `test -e/-f/-d/-x` checks;
3. verify Qwen from the real health endpoint;
4. verify SAM3 import origin and Boogu's separate runtime;
5. do not globally export `CUDA_VISIBLE_DEVICES`;
6. do not use broad `pkill` patterns;
7. treat public dataset/model roots as read-only;
8. use fresh run roots for changed identities;
9. write audit/replay outputs outside immutable source runs;
10. update this document whenever a path, GPU rule, service launch, production
    stage order, or validated baseline changes.
