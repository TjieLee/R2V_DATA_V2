# V3 Server Environment Runbook

Last updated: 2026-08-31

This is the authoritative server handoff for the frozen Visual/reference line
and the standalone Visual Stage2 Entity Mask production path. Verify the live
remote branch and HEAD before operating the server; do not infer server state
from old chat history.

## Freeze identity

```text
repository: TjieLee/R2V_DATA_V2

Visual/reference branch: feature/v3-subject-attributes-v1
final Visual/reference code freeze: d056c32b76db4b3d7c0358b38e996e7a91a288d1

standalone Entity Mask branch: feature/v3-sam3-production-v1
validated Entity Mask orchestration HEAD: 8645db869892ea98a6599467882efc48b4eb7414

frozen original Visual branch: feature/v3-runtime-integrity-v1
frozen original Visual HEAD: 87bd4e06107d7f56df550979b0e96515cb70f911
core original Visual algorithm baseline: 3cfb11fdd1fbe4a5bbad02a775097d8ab3097288
```

Docs-only commits may advance either branch HEAD without changing the Visual
algorithm freeze. Annotation production is frozen. Audio/H3 development remains
on its own line.

The standalone Entity Mask branch adds production orchestration around the
frozen Visual stages; it does not redefine the Visual/Audio production export
schema.

## Confirmed paths

```text
repo:
  /mnt/workspace/litengjie/data/R2V_DATA_V2

python:
  /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python

SAM3 code:
  /mnt/workspace/litengjie/data/vendor/sam3
SAM3 checkpoint:
  /mnt/workspace/public/pretrained/facebook/sam3/sam3.pt

Qwen model:
  /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct

Standalone Entity Mask candidate-judge gateway:
  http://6.167.57.88:8000/v1

Legacy/local full-V3 Qwen endpoint where applicable:
  http://127.0.0.1:8000/v1

Boogu code:
  /mnt/workspace/litengjie/data/vendor/Boogu-Image
Boogu python:
  /mnt/workspace/litengjie/data/venvs/boogu-image/bin/python
Boogu model:
  /mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708
```

Production data:

```text
source JSONL:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl
processed shot clips root:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped
original/full source videos root:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808

standalone Stage1 annotations:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations
standalone Stage2 Entity Mask output:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_mask
```

`video_path` is the processed Visual input shot, currently MP4.
`source_video_path` is provenance for the original/full video, is
container-extension agnostic, and must never be substituted for `video_path` as
model input.

## Full Visual/reference runtime allocation

The frozen full Visual/reference pipeline uses the following validated layout:

```text
GPU 0-3: Qwen3-VL-32B-Instruct, BF16, TP1 x DP4
GPU 4: Boogu background removal
GPU 5 + 7: shared two-process SAM3 pool
GPU 6: Boogu reference_edit and eligible non-face Attribute completion
GME: disabled

Qwen max model length: 49152
runtime.qwen_max_inflight: 4
OMP_NUM_THREADS: 1
```

The SAM3 pool serves main temporal segmentation and Attribute single-frame
probes. Boogu loads persistently; it is not reloaded per request. Fresh
Subject/Object and Attribute generated-background calls are zero. Reference
completion and Attribute completion continue to use GPU 6.

Do not apply this mixed GPU allocation to standalone Entity Mask production.
The standalone launcher reserves every visible local GPU for one persistent
SAM3 worker and uses the centralized remote Qwen candidate judge only when the
frozen ambiguity path requires it.

## Common shell environment

From the repository root:

```bash
export PYTHONPATH=/mnt/workspace/litengjie/data/vendor/sam3${PYTHONPATH:+:$PYTHONPATH}
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export OMP_NUM_THREADS=1
```

For the full Visual/reference pipeline, unset `CUDA_VISIBLE_DEVICES` unless the
specific launcher owns it:

```bash
unset CUDA_VISIBLE_DEVICES
```

Do not add Boogu or GME to `PYTHONPATH`. Boogu uses its configured interpreter
and code root. GME is disabled.

For multi-node Entity Mask production, `PYTHONPATH` is mandatory on every
node. The same `.venv/bin/python` path alone does not make the vendored SAM3
package importable. Before launching, run on every node:

```bash
/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python - <<'PY'
from sam3.model_builder import build_sam3_video_predictor
print("SAM3_IMPORT_OK")
PY
```

Every node must print `SAM3_IMPORT_OK`. The validated multi-node failure mode for
a missing environment entry was:

```text
ModuleNotFoundError: No module named 'sam3'
```

## Safe repository update

Preserve all untracked server-local files. Do not run `git clean`, reset,
rebase, or force-push operations.

For Visual/reference work:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
git fetch origin
git switch feature/v3-subject-attributes-v1
git merge --ff-only origin/feature/v3-subject-attributes-v1
git status --short
git rev-parse HEAD
```

For standalone Entity Mask production:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
git fetch origin
git switch feature/v3-sam3-production-v1
git merge --ff-only origin/feature/v3-sam3-production-v1
git status --short
git rev-parse HEAD
```

The branch HEAD may be a docs-only descendant of a validated code commit.
Inspect the commit subjects before treating a newer HEAD as an algorithm change.

## Qwen service

The full Visual/reference pipeline may run the local Qwen3-VL-32B-Instruct
service with:

```text
model: /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
dtype: bfloat16
tensor parallel size: 1
data parallel size: 4
visible GPUs: 0,1,2,3
max model length: 49152
endpoint: http://127.0.0.1:8000/v1
```

Standalone Entity Mask does not start local Qwen workers on its SAM3 nodes. Its
frozen targeted ambiguity judge uses:

```text
http://6.167.57.88:8000/v1
/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
```

The Entity Mask launcher performs a service health check before spawning SAM3
workers.

## Standalone Entity Mask production

Formal production input/output:

```text
input:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations

output:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_mask

config:
/mnt/workspace/litengjie/data/entity_mask_configs/production.yaml

logs:
/mnt/workspace/litengjie/data/entity_mask_logs
```

External entry:

```bash
bash stage_entity_mask_run_v2.sh
```

`RANK` and `WORLD_SIZE` come from the cluster environment. The launcher detects
all visible local GPUs. The exact node count is not hard-coded.

Parallel ownership is:

```text
global_worker_count = WORLD_SIZE * local_gpu_count
global_worker_id    = RANK * local_gpu_count + local_gpu_slot
```

Each complete 10,000-row Stage1 shard belongs to exactly one global GPU worker.
Assignments are contiguous, balanced, deterministic, and disjoint. A worker
loads SAM3 once and processes all complete shards in its fixed range.

Inside each 10k logical shard, the existing 100-row durable chunks remain for
checkpoint/resume and final canonical compaction. They are recovery units, not
cross-GPU work units.

The formal static path does not use metadata, frame, chunk-claim, or compaction
`flock` during normal data processing. Rank 0 still uses `execution.lock` only
for startup execution/session identity initialization. Generic dynamic Stage2
runners keep their old lock behavior.

The 2026-08-29 dynamic multi-node attempt exposed workspace-level missing-file
failures involving `run.json` and `masks.rle.json`. The exact distributed
filesystem mechanism was not proven. Formal Entity Mask production therefore
moved from cross-node dynamic work stealing to static complete-shard ownership
so two nodes never intentionally operate the same shard/workspace.

Startup identity includes the exact `WORLD_SIZE`, local GPU count, global worker
count, ordered shard-list hash, chunk size, and session-reuse mode. Rank > 0
retries only temporary `FileNotFoundError` visibility for up to 120 seconds;
identity/schema mismatch fails immediately.

Do not reuse a single-node smoke root for a later multi-node topology. Archive
the whole smoke output root and let formal production initialize a fresh
`entity_mask` root.

The accepted throughput settings are:

```text
execution chunk rows:       100
frame prefetch workers:     0
SAM3 request timing:        disabled
SAM3 session reuse:         clip_reset_v1
debug diagnostics:          disabled
binary-mask RLE:            vectorized NumPy implementation
```

The measured Stage2 optimization sequence improved the same 80-row workload
from `5m11.471s` to `1m43.285s`, approximately 66.8% lower wall time and 3.02x
throughput. Real-mask A/B comparison found zero differences in 54
`masks.rle.json` files. Canonical Stage2 JSONL was subsequently confirmed
byte-identical between the compared scalar/vectorized execution paths, with
metadata differing only by expected timestamps.

See:

```text
docs/ENTITY_MASK_PRODUCTION.md
docs/V3_STAGE2_E2E_PERFORMANCE_OPTIMIZATIONS.md
```

## Entity Mask stop/restart

Terminate the node launcher first:

```bash
pkill -TERM -f 'run_v3_entity_mask_auto.py'
```

The launcher terminates and reaps only its own workers, using a shared 10-second
grace period before `SIGKILL` escalation. If the launcher itself was externally
killed with `SIGKILL`, use this only as a fallback:

```bash
pkill -TERM -f 'run_v3_entity_mask_worker.py'
```

A worker ordinary failure does not fail-fast other fixed workers; healthy
workers finish their assigned shards and the launcher returns a nonzero final
status if any child failed.

## Fresh full Visual/reference run

The exact full stage order is:

```text
manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,subject_attributes,export
```

Set task-specific paths, then create only the config parent directory:

```bash
export CONFIG=/absolute/path/to/config.yaml
export RUN_ROOT=/absolute/path/to/new-run-root
export EXPORT_ROOT=/absolute/path/to/new-export-root
mkdir -p "$(dirname "$CONFIG")"
```

Do not precreate `RUN_ROOT`. In particular, do not precreate `EXPORT_ROOT`.
`DatasetExporter` requires the destination not to exist and publishes through a
temporary directory atomically.

Launch:

```bash
.venv/bin/python run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,subject_attributes,export \
  --profile
```

If an existing run identity does not match the requested configuration or
model identity, use a new run root. Do not delete `run.json` and do not weaken
identity validation.

## Frozen selection policies

Subject/Object:

```text
complete or local_usable -> canonical alpha
repairable -> candidate 1 completion and comparative Qwen
  -> reject/not better: candidate 2 when available
  -> reject/not better: canonical alpha
selected completion integrity failure -> canonical alpha integrity
canonical alpha integrity failure -> bbox last resort
```

Completion never falls directly to bbox. Source-relative area, scale, and
center changes are diagnostic only for completion; identity, duplicate/wrong
instance, fragmentation, tiny/extreme placement, and severe warping remain
hard gates.

Attributes:

```text
at most 3 per eligible human owner
owner Top3 candidates; at most 2 different sources
single-frame SAM3 probes only; no temporal tracking
6 hard raw review flags, including sufficient_source_evidence
structure_complete and completion_recommended are diagnostics only
face: Boogu hard-disabled, accepted raw -> reviewed bbox -> raw alpha fallback
face without an accepted raw candidate -> reject; bbox cannot bypass raw gates
eligible non-face repair uses raw alpha + source RGB bbox + Boogu candidate
non-face insufficient evidence -> candidate 2, then bbox last resort
non-face bbox reject -> Attribute reject
fresh generated background -> disabled
```

See `V3_REFERENCE_EDIT_BOOGU.md` and `V3_SUBJECT_ATTRIBUTES_STATE.md` for exact
prompts and gates.

## Final validation evidence

Latest fixed-100 real-model run:

```text
run: e2e100-verify-324a29a-20260825-234558
commit used: 324a29aebcf4b573cab59332d337ec9d10ad9deb
```

```text
reference_edit background attempted/accepted: 0/0
reference_edit completion attempts: 5
candidate 1 accepted: 3
candidate 2 attempts: 0
fallback alpha: 2
reference_edit entities accepted/rejected: 95/0

reference_integrity entities accepted/rejected: 84/17
reference_integrity bbox fallback attempted/accepted: 1/1

attributes accepted: 84
attribute completion attempts/accepted: 70/51
attribute Qwen completion rejects: 19
attribute second candidate attempts: 21
attribute candidate 1/candidate 2 accepted: 46/5
attribute bbox fallback attempted/accepted: 3/0
attribute background variants: 0/0
```

The later validator fix, prompted by a one-owner/three-attribute batch-loss bug,
passed the full local suite but did not rerun this fixed-100 real-model canary.
The candidate-2 source-frame provenance fix at the earlier `d7f3d6b...` freeze
was validated locally with 368 focused and 1,991 full tests passing (one
warning), diff-check, and compileall. It also did not run a new GPU/model canary.

The current face bbox-first policy was validated locally with 148 focused and
2,000 full tests passing (one warning), compileall, and diff-check. Ruff found
the same 69 pre-existing issues on the clean parent and updated tree, so this
change introduced no new Ruff finding. No Qwen, SAM3, Boogu, GPU, or real-model
canary was run for it.

## Audio/H3 handoff

For cross-branch integration, consume compacted
`r2v.v3.production_sample.1`. Do not treat internal Attribute owner/review
sidecars or standalone Entity Mask internal checkpoints as the final
integration API.

The recent standalone Entity Mask commits are isolated Visual Stage2
orchestration changes. They do not modify `r2v_data_v2/h3/*` or the Audio/H3
branch unless explicitly merged/cherry-picked. Read
`V3_VISUAL_AUDIO_INTEGRATION.md` before updating Audio/H3.
