# R2V V3 Server Environment Runbook

Last updated: 2026-08-20

This file is the single source of truth for the Linux server environment used by
the current integrated Visual V3 + subject-attribute pipeline. Update it whenever
a runtime, model path, service endpoint, GPU assignment, launch command, or
isolation rule changes. Do not reconstruct these values from chat history.

`CONFIRMED` means the value is backed by repository/server evidence. `UNVERIFIED`
means it must be checked on the server before use. Do not replace an unverified
value with a guess.

Current integrated Visual + subject-attribute branch and validated code baseline:

```text
branch: feature/v3-subject-attributes-v1
code baseline: 06051245ea4f58ca8b1df5aa117fab918f211533
```

Frozen Visual baseline remains separate:

```text
branch: feature/v3-runtime-integrity-v1
frozen HEAD: 87bd4e06107d7f56df550979b0e96515cb70f911
```

Subject attributes are a sidecar extension and must not silently change the
frozen Visual export/sample semantics or Visual thresholds. Documentation-only
commits may move repository HEAD after the validated code baseline; record both
when reproducing or freezing a run.

For the detailed attribute contract and benchmark state, see
`V3_SUBJECT_ATTRIBUTES_STATE.md`. Historical frozen Visual integrity evidence
remains in `V3_RUNTIME_INTEGRITY_STATE.md`.

## 1. Runtime Layers

```text
PRODUCTION
  main R2V .venv
    - V3 streaming orchestration
    - persistent isolated GPU workers
    - OpenAI-compatible Qwen client
  external service
    - vLLM serving Qwen3-VL-32B-Instruct
  separate runtime
    - Boogu Image worker

PERSISTENT GPU WORKERS
  physical GPU 4
    - Object Remover
  physical GPU 5
    - main temporal SAM3
  physical GPU 6
    - reference edit / Boogu worker path as configured
  physical GPU 7
    - subject-attribute single-frame SAM3

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

JEA production source JSONL:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl

JEA processed clips_root:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped

JEA original source_videos_root:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808

Boogu code:
  /mnt/workspace/litengjie/data/vendor/Boogu-Image

Boogu Python:
  /mnt/workspace/litengjie/data/venvs/boogu-image/bin/python

Boogu model:
  /mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708

Object-Remover LoRA:
  /mnt/workspace/litengjie/data/models/Qwen-Image-Edit-2511-Object-Remover/Qwen-Image-Edit-2511-Object-Remover.safetensors
```

For JEA production, `record.video_path` is the existing processed MP4 shot
below `clips_root` and is the only one of these two fields used as V3 Visual
frame/model input. `record.source_video_path` is original/full-video provenance
only: it must be a safe existing regular file below `source_videos_root`, but
its container and extension are unrestricted. For example,
`01/丁宝桢/01 4K.mkv` is valid provenance. Never substitute
`source_video_path` for `video_path` as Visual input.

Writable roots:

```text
/mnt/workspace/litengjie/data/r2v_v3_configs
/mnt/workspace/litengjie/data/r2v_v3_runs
/mnt/workspace/litengjie/data/r2v_v3_exports
/mnt/workspace/litengjie/data/r2v_v3_logs
/mnt/workspace/litengjie/data/r2v_v3_audits
/mnt/workspace/litengjie/data/r2v_v3_reviews
/mnt/workspace/litengjie/data/r2v_v3_subject_attributes
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

## 3. Production GPU Allocation

Current validated allocation:

```text
physical GPU 0  Qwen3-VL DP replica 0, TP1
physical GPU 1  Qwen3-VL DP replica 1, TP1
physical GPU 2  Qwen3-VL DP replica 2, TP1
physical GPU 3  Qwen3-VL DP replica 3, TP1
physical GPU 4  Object Remover
physical GPU 5  main temporal SAM3
physical GPU 6  Boogu / reference-edit worker
physical GPU 7  dedicated subject-attribute single-frame SAM3
```

Do not globally export `CUDA_VISIBLE_DEVICES` in the normal parent pipeline
shell. The parent runtime assigns physical GPUs to isolated workers. Before
launching the full pipeline, use:

```bash
unset CUDA_VISIBLE_DEVICES
```

Important ordinal rules:

- main SAM3 uses physical GPU5 through worker isolation and YAML
  `sam3.device: cuda`; never put `cuda:5` in the worker-local SAM3 config;
- dedicated attribute SAM3 uses physical GPU7 through the same isolation rule
  and also uses worker-local `cuda`;
- Object Remover is assigned physical GPU4 by runtime GPU-worker configuration;
- Boogu is assigned physical GPU6 and sees its isolated local CUDA device as
  `cuda:0`;
- Qwen is an external vLLM service on physical GPUs 0-3 and is not launched by
  `run_pipeline_v3.py`.

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

The production subject-attribute path runs from the integrated branch, not the
Audio/H3 branch and not the frozen Visual branch.

```bash
cd "$R2V_REPO_ROOT"

git status --short
git fetch origin feature/v3-subject-attributes-v1
git switch feature/v3-subject-attributes-v1
git pull --ff-only origin feature/v3-subject-attributes-v1

git rev-parse HEAD
git branch --show-current
git status --short
```

Expected validated code baseline before documentation-only commits:

```text
06051245ea4f58ca8b1df5aa117fab918f211533
```

If `git status --short` is non-empty before switching, stop. Do not stash,
reset, force-switch, or overwrite another branch's changes.

Fixed-path checks:

```bash
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

Verify Boogu separately without inheriting production `PYTHONPATH`:

```bash
env -u PYTHONPATH "$BOOGU_PYTHON" - <<'PY'
import sys
import torch
print('Python', sys.version)
print('torch', torch.__version__)
PY
```

## 6. Qwen3-VL / vLLM

Current production serving baseline:

```text
model:                /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
endpoint:             http://127.0.0.1:8000/v1
dtype:                BF16
tensor parallel:      1
data parallel:        4
max model length:     49152
pipeline max inflight: 4
```

The validated deployment direction is four independent TP1 replicas on physical
GPUs 0-3 behind one vLLM endpoint. This replaced TP4 because the full streaming
10-clip canary improved from about 889 seconds to about 327 seconds.

Current launch shape:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=1

vllm serve /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --max-model-len 49152 \
  --allowed-local-media-path /mnt/workspace/public/dataset \
  --port 8000
```

This block records the required validated topology. Before restarting a live
server, inspect `ps -ef` and preserve any machine/version-specific flags already
verified on that server. Do not silently revert to TP4 or change dtype/FP8 as
part of an unrelated pipeline run.

Health check:

```bash
curl -fsS --noproxy '*' http://127.0.0.1:8000/v1/models
```

Inspect the real service before terminating anything:

```bash
ps -ef | grep '[v]llm'
ps -ef | grep '[a]pi_server'
```

Terminate only the specific verified PID with `kill -TERM`. Never use a broad
`pkill -f python`.

The pipeline-level global Qwen gate is:

```yaml
runtime:
  qwen_max_inflight: 4
```

The gate scans all global file-lock slots and takes the first available slot
from a rotating start point. Profiling model-call metadata includes `qwen_slot`
for diagnosis.

## 7. Frozen Visual Contract + Sidecar Runtime Settings

The frozen Visual semantics must remain unchanged by subject-attribute work:

```yaml
source:
  selection_mode: fixed_selection_v1

sam3:
  device: cuda
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

reference_edit:
  scale_collapse_fallback_guard_mode: qwen_v1

reference_integrity:
  enabled: true
  mode: targeted_qwen_v1
```

Current integrated runtime-capacity settings:

```yaml
runtime:
  mode: streaming_v1
  qwen_max_inflight: 4
  stage_workers:
    subject_attributes: 2
  gpu_workers:
    segment: "5"
    subject_attributes_segment: "7"
    remove: "4"
    reference_edit: "6"
```

`runtime.stage_workers.subject_attributes` and
`runtime.gpu_workers.subject_attributes_segment` are sidecar runtime-capacity
settings and are excluded from the frozen Visual fingerprint/model identifiers.
The Git commit remains part of run identity.

Same-parent cross-pair remains disabled for the frozen Visual path. Do not loosen
Visual thresholds to increase export yield.

## 8. Subject-Attribute Runtime Contract

Attribute enrichment is precision-first and fail-closed. Key production limits:

```text
maximum discovered attributes per owner: 3
Qwen discovery calls: <= 1 per eligible human owner
SAM3 attribute mode: single frame only
candidate frames: owner's existing Top3 only
attribute temporal tracking: disabled
attribute 7/10 coverage rule: not used
attribute generative completion: not used
```

Accepted attribute crops are source RGB plus mask exported as RGBA transparency.
Every accepted record carries `owner_entity_id` and must pass deterministic
ownership geometry plus strict batched Qwen recognizability/owner-binding review.

Attribute outputs are sidecars below `<run_root>/subject_attributes/`; they do
not mutate the frozen Visual `ClipRecord` or frozen Visual export schema.

See `V3_SUBJECT_ATTRIBUTES_STATE.md` for the full contract and current evidence.

## 9. Full Stage Order and Launch

Current integrated full stage order:

```text
manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,subject_attributes,export
```

Do not use old snippets that omit `reference_integrity` or
`subject_attributes` when running the integrated production branch.

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

Before launch, verify at minimum:

```text
runtime.mode = streaming_v1
runtime.qwen_max_inflight = 4
runtime.gpu_workers.segment = "5"
runtime.gpu_workers.subject_attributes_segment = "7"
runtime.gpu_workers.remove = "4"
runtime.gpu_workers.reference_edit = "6"
sam3.device = cuda
```

Launch a fresh full run:

```bash
unset CUDA_VISIBLE_DEVICES

nohup "$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,subject_attributes,export \
  --profile \
  > "$LOG" 2>&1 &

echo $!
```

Use a fresh run root. `run.json` binds run ID, Git commit, config hash, model
identifiers, and source manifest path. Never weaken this identity check.

## 10. Bounded Replays

Prefer the smallest replay that can validate a change.

Main SAM3 rescue A/B when annotation/frames are already prepared:

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

Attribute-only integrated streaming replay is allowed when the required upstream
artifacts are already valid and the fresh/current run identity matches:

```bash
"$MAIN_PYTHON" run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages subject_attributes \
  --profile
```

If `subject_attributes_segment` is configured, this starts the dedicated GPU7
segment worker for attribute probes. Without it, the code falls back to the main
segment worker.

## 11. Monitoring

```bash
pgrep -af 'run_pipeline_v3.py'

nvitop
nvidia-smi

tail -n 100 "$LOG"
tail -f "$LOG"
```

Expected persistent-worker stderr logs include:

```text
$RUN/logs/streaming_segment_worker.stderr.log
$RUN/logs/streaming_subject_attributes_segment_worker.stderr.log
$RUN/logs/streaming_remove_worker.stderr.log
$RUN/logs/streaming_reference_edit_worker.stderr.log
```

Profiling files:

```text
$RUN/profiling/events.jsonl
$RUN/profiling/summary.json
```

Subject-attribute outputs:

```text
$RUN/subject_attributes/owners/
$RUN/subject_attributes/references/
$RUN/subject_attributes/samples/
$RUN/subject_attributes/attributes.jsonl
$RUN/subject_attributes/enriched_samples.jsonl
$RUN/subject_attributes/summary.json
```

A missing profiling file may simply mean the run was not launched with
`--profile`; it is not alone proof of model failure.

## 12. Current Evidence Paths

2026-08-20 new-data 10-clip functional canary:

```text
commit:                       06051245ea4f58ca8b1df5aa117fab918f211533
source range:                  0-9
input clips:                   10
Visual exports:                3
Visual references:             4
eligible human owners:         2
accepted attribute references: 2
enriched samples:              2
canonical samples:             3
canonical total references:    6
failed_tasks =                 []
copied videos:                 0
```

The TP1 x DP4 BF16 service on physical GPUs 0-3 used
`--max-model-len 49152`, the pipeline retained `runtime.qwen_max_inflight: 4`,
and `OMP_NUM_THREADS=1`. The 49152 setting eliminated the previous 32768
context-length infrastructure failure. This is bounded functional evidence;
3 exports from 10 inputs must not be treated as a production yield estimate.

Frozen Visual 1000-clip canary:

```text
run: /mnt/workspace/litengjie/data/r2v_v3_runs/e2e1000-s0-samfix-20260814-101818
config: /mnt/workspace/litengjie/data/r2v_v3_configs/e2e1000-s0-samfix-20260814-101818.yaml
export: /mnt/workspace/litengjie/data/r2v_v3_exports/e2e1000-s0-samfix-20260814-101818
```

Observed frozen Visual export:

```text
raw clips: 1000
exported samples: 362
references: 530
yield: 36.2%
```

Integrated TP4 subject-attribute canary:

```text
/mnt/workspace/litengjie/data/r2v_v3_runs/stream-attr10-b0790e-20260819-211823
```

TP1 x DP4 benchmark:

```text
/mnt/workspace/litengjie/data/r2v_v3_runs/stream-attr10-dp4-889b45-20260819-221452
```

Current dedicated-GPU7 benchmark:

```text
/mnt/workspace/litengjie/data/r2v_v3_runs/stream-attr10-dp4-sam7-c1c056-20260819-223752
```

Do not compare the two 10-clip attribute runs as identical compute paths; model
responses changed some downstream workload. The validated large performance
change is TP4 -> TP1 x DP4. The GPU7 worker is retained for resource isolation.

## 13. Failure Signatures

### `existing run.json does not match the requested V3 run`

Commit, config hash, model identifiers, manifest, or run ID differs. Use a fresh
run root rather than weakening validation.

### SAM3 worker reports invalid device / CUDA ordinal

For isolated GPU5/GPU7 workers, YAML must use `sam3.device: cuda`, not
`cuda:5`/`cuda:7`. Physical selection comes from runtime GPU-worker assignment.

### Attribute probes unexpectedly block main temporal SAM3

Check:

```yaml
runtime:
  gpu_workers:
    segment: "5"
    subject_attributes_segment: "7"
```

Then inspect both persistent-worker logs. If the optional GPU7 assignment is
absent, fallback to the main segment worker is expected behavior.

### Qwen queue wait grows sharply

Confirm the external service is still TP1 x DP4 and the pipeline config still
uses `runtime.qwen_max_inflight: 4`. Profiling model-call metadata includes
`queue_wait_seconds`, `qwen_inflight`, and `qwen_slot`.

### Qwen integrity JSON EOF / schema error

Current code retries one structured-output repair. If both attempts fail, the
entity fails closed and diagnostics retain raw responses/finish reasons. Do not
route a judge failure into bbox fallback.

### Historical clip fails newest schema

For read-only evidence extraction, parse only the stable sections required for
the task. For a replay that writes current artifacts, use a fresh/current run
root with consistent downstream state; do not delete `run.json` to bypass
identity checks.

## 14. Known Subject-Attribute Resume Edge Cases

Two unusual resume/reconciliation cases remain tracked:

1. If upstream rerun changes a previously eligible clip to export-ineligible,
   stale durable attribute sidecars can remain unless explicitly invalidated by
   overwrite/reconciliation.
2. An owner exception outside durable owner-artifact creation can be visible in
   the current runtime invocation but undercounted when the aggregate summary is
   later rebuilt only from durable owner artifacts.

These were not observed in the fresh production canaries. They are not a reason
to relax attribute quality gates.

## 15. Audit-Only Isolation

The reference-filter/pose environment remains isolated from production:

```text
/mnt/workspace/litengjie/data/venvs/r2v-reference-filter/bin/python
```

Use `env -u PYTHONPATH` for pose/SCRFD/MediaPipe commands. Do not install SAM3
there. DINO/SigLIP/SCRFD/MediaPipe observations remain audit evidence and do not
become production thresholds without an explicit reviewed change.
