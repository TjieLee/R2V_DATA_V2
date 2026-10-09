# Post-Mask Resource Epoch Production

On each 8-GPU node, from the repository checkout:

```bash
bash scripts/run_v3_post_mask_full_production.sh
```

For a path/config review without loading models or requiring server dependencies:

```bash
bash scripts/run_v3_post_mask_full_production.sh --dry-run
```

The launcher accepts externally assigned `RANK` and `WORLD_SIZE` (default
`0/1`). They rotate scan order only. Every node sees every canonical group and
claims unfinished work via a nonblocking shared `flock`. A node that encounters
busy groups waits and scans again. A crash releases the lock; a later launch can
resume with a different node count. Canonical per-shard locks remain in force.

The frozen Stage2 input is
`/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_mask`.
The direct DatasetExporter destination and aggregate production output are under
`/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference`.
Stable group/shard state and locks live in `in_pair_reference/state`; private
intermediate RunStorage and model scratch remain under
`/mnt/workspace/litengjie/data/r2v_v3_runs/production/jea_motion_v1/in_pair_reference`.
No timestamped output root or copy-back step is used.

Each shard gets a small atomic `completed.json` only after the existing
Subject Attributes terminal barrier and DatasetExporter publication. The group
marker follows only after every shard export succeeds. Restart skips completed
shards/groups by reading those markers, without rehashing export trees, receipts,
or media. Missing markers enter the existing durable ModelJob receipt replay.
Once all groups finish, one node holds the compaction lock and uses the existing
V3 compactor to publish `samples.jsonl` and `catalog.json` at the official root.
The compactor's normal publication checks are not used as a completion/resume
gate. A plain compaction marker makes later launches skip that work as well.

The committed config pins Boogu removal and all Qwen services to
`Qwen3-VL-8B-Instruct`; the launcher pins the managed physical Qwen model after
loading `server_env.sh`. Existing Resource Epoch resource sessions own 8 Boogu,
8 SAM and TP1/DP8 Qwen GPU slots in sequence, not concurrently. No Audio/H3
stage is launched here.

Server-local checkpoint availability, shared-filesystem `flock`, write access
to the public output root, and real model quality still require an on-server
small smoke before full production. This code change runs no real models.

## Isolated single-node, 8-GPU A/B smoke (operator-run only)

Compare baseline `bc3f59f78df8db9b3d454983695e787998f60a17` with the delivered
immutable P0-fix commit on `feature/post-mask-sa-sam-pipeline-v1`. The commands
below have not been run on the server; no GPU throughput improvement is measured.

Supply the same accepted frozen config and an **existing frozen Entity Mask
fixture containing only 30–100 clips** for both arms. The generic launcher reads
every canonical `parts/shard-*.jsonl` in that root: `source.limit` and
`selection_manifest` do not restrict it. Never use the full production Entity
Mask root for this smoke. Preserve all accepted semantic settings/checkpoints;
do not invent a config, seed plan, or fixture here. For exact generated-output
parity, use an already accepted fixture with matching frozen upstream references
and generation seeds under each correct run identity. Do not copy completed
workspaces across roots or rewrite their identities.

Do not invoke `run_v3_post_mask_full_production.sh`, edit/fetch/check out in
`/mnt/workspace/litengjie/data/R2V_DATA_V2_postmask_prod`, or touch formal
production state/exports. The following clones are independent; all smoke
writes, caches, scratch and logs stay below `/mnt/workspace/litengjie/data`.
Public source/model trees remain read-only. Use the existing runtime unchanged;
do not install packages or download weights.

Both arms use the same accepted config: Boogu/SAM checkpoint/runtime paths are
inherited unchanged, while Qwen is explicitly fixed below to
`Qwen3-VL-8B-Instruct`. Existing resource defaults are Qwen TP1/DP8 and eight
Boogu/SAM workers (one per physical GPU 0–7), loaded in separate resource epochs.

In a fresh Bash shell, supply the three required operator values. Use unused
checkout paths and output tags below; if they already contain a prior smoke,
choose new names for a new comparison (resume must retain the original names).

```bash
SA_NEW_COMMIT=${SA_NEW_COMMIT:?Set the delivered immutable P0-fix commit SHA}
SMOKE_CONFIG=${SMOKE_CONFIG:?Set the accepted frozen smoke config path}
SMOKE_ENTITY_MASK_ROOT=${SMOKE_ENTITY_MASK_ROOT:?Set the existing 30-100-clip fixture root}
SA_BASELINE_COMMIT=bc3f59f78df8db9b3d454983695e787998f60a17
SA_BASELINE_REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2_sa_p0_baseline
SA_NEW_REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2_sa_p0_new
SA_PYTHON=/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python
SA_SERVER_ENV=/mnt/workspace/litengjie/data/R2V_DATA_V2/server_env.sh
SA_BASELINE_TAG=sa-p0-baseline-smoke-r1
SA_NEW_TAG=sa-p0-new-smoke-r1

# Fresh independent clones only; no Git operation in a production checkout.
test -r "$SMOKE_CONFIG" && test -d "$SMOKE_ENTITY_MASK_ROOT/parts" || exit 1
test -x "$SA_PYTHON" && test -r "$SA_SERVER_ENV" || exit 1
test ! -e "$SA_BASELINE_REPO" && test ! -e "$SA_NEW_REPO" || exit 1
git clone --no-checkout https://github.com/TjieLee/R2V_DATA_V2.git "$SA_BASELINE_REPO" || exit 1
git clone --no-checkout https://github.com/TjieLee/R2V_DATA_V2.git "$SA_NEW_REPO" || exit 1
git -C "$SA_BASELINE_REPO" fetch origin feature/post-mask-sa-sam-pipeline-v1 || exit 1
git -C "$SA_NEW_REPO" fetch origin feature/post-mask-sa-sam-pipeline-v1 || exit 1
git -C "$SA_BASELINE_REPO" checkout --detach "$SA_BASELINE_COMMIT" || exit 1
git -C "$SA_NEW_REPO" checkout --detach "$SA_NEW_COMMIT" || exit 1
git -C "$SA_BASELINE_REPO" rev-parse HEAD || exit 1
git -C "$SA_NEW_REPO" rev-parse HEAD || exit 1
git -C "$SA_BASELINE_REPO" status --short || exit 1
git -C "$SA_NEW_REPO" status --short || exit 1

run_sa_smoke() (
  source "$SA_SERVER_ENV" || return 1
  repo="$1"; tag="$2"; mode="$3"
  root="/mnt/workspace/litengjie/data/r2v_v3_post_mask/jea_motion_v1/$tag"
  cd "$repo" || return 1
  export HF_HOME=/mnt/workspace/litengjie/data/cache/huggingface
  export TORCH_HOME=/mnt/workspace/litengjie/data/cache/torch
  export XDG_CACHE_HOME=/mnt/workspace/litengjie/data/cache/xdg
  export TMPDIR=/mnt/workspace/litengjie/data/tmp
  mkdir -p "$root" "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" || return 1
  if [ "$mode" = dry ]; then
    args=(--dry-run)
  else
    args=(--job-runner r2v_data_v2.v3.post_mask_epoch_removal:run_removal_epoch)
  fi
  env -u POST_MASK_FORMAL_PRODUCTION \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 \
    NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}" \
    no_proxy="127.0.0.1,localhost,${no_proxy:-}" \
    PYTHONPATH="$repo:/mnt/workspace/litengjie/data/vendor/sam3${PYTHONPATH:+:$PYTHONPATH}" \
    POST_MASK_REPO="$repo" POST_MASK_CPU_WORKERS=32 \
    POST_MASK_QWEN_MAX_INFLIGHT=8 \
    POST_MASK_QWEN_MODEL_PATH=/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct \
    POST_MASK_EPOCH_TEMP_ROOT="$root/tmp/resource_epoch" \
    "$SA_PYTHON" "$repo/tools/run_v3_post_mask_resource_epoch.py" \
      --base-config "$SMOKE_CONFIG" --entity-mask-root "$SMOKE_ENTITY_MASK_ROOT" \
      --post-mask-root "$root" --tag "$tag" \
      --rank 0 --world-size 1 --group-size 8 "${args[@]}" \
      2>&1 | tee -a "$root/smoke-$mode.log"
  return "${PIPESTATUS[0]}"
)

# Model-free planning first; this writes only private descriptors/locks/summary.
run_sa_smoke "$SA_BASELINE_REPO" "$SA_BASELINE_TAG" dry || exit 1
run_sa_smoke "$SA_NEW_REPO" "$SA_NEW_TAG" dry || exit 1
```

Review both logs before proceeding: commits, config, fixture scope, canonical
shards/group count and all destinations must match the intended isolated smoke.
Dry-run does not load the runner/models and does not itself validate model
availability or report per-shard run/export destinations. With formal production
unset, `ShardPaths.for_shard` derives these destinations from the basename of
`--post-mask-root` (the tag above), overriding the base config's physical
run/export placement:

| Artifact | Exact path under `/mnt/workspace/litengjie/data` |
| --- | --- |
| State/group summary | `r2v_v3_post_mask/jea_motion_v1/<tag>/resource_epochs/summary-rank0.json` |
| Shard state/lock | `r2v_v3_post_mask/jea_motion_v1/<tag>/shards/<shard>/` |
| Private RunStorage | `r2v_v3_runs/production/jea_motion_v1/<tag>/<shard>/` |
| Private DatasetExporter | `r2v_v3_exports/production/jea_motion_v1/<tag>/shards/<shard>/` |
| Model scratch/logs | `r2v_v3_post_mask/jea_motion_v1/<tag>/tmp/resource_epoch/` |

Despite the historical `production` directory component, the unique smoke tags
are separate from the formal `in_pair_reference` campaign. Check that these
destinations are unused for each fresh arm. Then, on the server only, execute
sequentially on the same eight idle GPUs; never run both arms concurrently or
compete with a production campaign:

```bash
run_sa_smoke "$SA_BASELINE_REPO" "$SA_BASELINE_TAG" run || exit 1
run_sa_smoke "$SA_NEW_REPO" "$SA_NEW_TAG" run || exit 1
# Resume: same checkout, config, fixture, tag and roots, without overwrite/delete.
run_sa_smoke "$SA_NEW_REPO" "$SA_NEW_TAG" run || exit 1
```

For interruption testing, interrupt only the isolated new smoke, then rerun its
exact command. Compare completion/skip counts, completed jobs/owners/attributes,
accepted outputs and failures (`smoke-run.log`, group summaries and shard
`failures.jsonl`). A completed group is skipped before loading its runner; an
unfinished group uses existing durable receipts. Do not delete state to test
resume. Generated-byte equality requires the matching pre-frozen seeds above.

Measurement: use `post_mask_epoch_stage_scheduler` with
`stage=subject_attributes`, `resources.sam.epoch_wall_seconds`, summed over SA
SAM waves, and report completed clips/owners/attributes alongside it. This
loaded-wave wall time excludes SAM startup but includes prepare/persist/CPU
settlement; neither whole-pipeline wall time nor model residence is pure GPU
time. Keep stage seed/scheduler/reconcile timing, resource
`startup_wall_seconds`/`service_seconds`, `settle_wall_seconds` and
`owner_replay_wall_seconds` separate. The new execution diagnostics event
`post_mask_epoch_subject_attributes_sam_execution_diagnostics` exposes
`infer_seconds`, `persist_seconds`, `persist_wait_seconds`, overlap, per-slot
inference time and occupancy/capacity. `model_call_time_seconds` is inference
wall plus actual persistence wall, **excluding asynchronous writer-queue wait**;
it is not pure GPU compute. Baseline diagnostics may lack the new fields.

Stop after reviewing the smoke and resume results. This procedure does not
authorize deployment, merge, production restart or changes to formal outputs.

## Fast SA restart priority + 4 SAM processes per GPU (experimental)

This section applies **only** to
`feature/post-mask-sa-4proc-priority-io-v1`. The code has NOT passed an
end-to-end 8-GPU SA production-resume smoke, and must not be installed into
the live shared `R2V_DATA_V2_postmask_prod` worktree while any node runs.

**Why:** the isolated 256-real-prompt H200 benchmark measured
1.587 / 2.621 / 3.819 SAM segment calls/s at 1 / 2 / 4 *independent CUDA
processes* on ONE GPU (2.407x four-versus-one). This benchmark deliberately
excluded SA NPY disk persistence, Qwen/Boogu model co-residency, receipt
settlement and CPU finalizers. Of 32 sampled tasks, 4-process mode returned
40/40 identical masks; 2-process mode had one pixel-mismatching mask, which is
recorded as a performance-experiment observation rather than turned into a
new costly validation gate. Neither result proves full semantic equivalence.

### Operator action BEFORE an elastic restart

Generate one shared snapshot on one node. This reads only existing tiny
`composition/subject_attributes_*.json` metadata plus a **single**
`os.scandir` count of terminal outcome filenames in each started-SA group.
It does not recurse through masks, images, model runs or receipts, and does not
create or modify a production checkpoint. Run this step ONCE, not on every
node. All paths below are on the shared filesystem.

```bash
DEV=/mnt/workspace/litengjie/data/R2V_DATA_V2_sa_4proc_dev
PY=/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python
PRIORITY=/mnt/workspace/litengjie/data/r2v_v3_sa_group_priority_20261009.json
STATE=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference/state/resource_epochs

"$PY" "$DEV/tools/prepare_v3_post_mask_sa_priority.py" \
  --state-root "$STATE" --output "$PRIORITY"
```

The printed order is: SA completed/export pending first, then SA started
descending terminal/eligible clip fraction, then other groups in original
order. Groups already complete are always skipped. On next **independently
approved, pinned, all-nodes-restarted** formal launch, set the same environment
on **every node** (after any server-env overrides, before the launcher):

```bash
export POST_MASK_SA_GROUP_PRIORITY_FILE="$PRIORITY"
export POST_MASK_SA_SAM_PROCESSES_PER_GPU=4
export POST_MASK_SA_SAM_PREPARE_WORKERS=8
export POST_MASK_SA_SAM_PERSIST_WORKERS=8
```

When the priority flag is unset, legacy rank rotation is unchanged. When set,
all ranks try the same shared priority order and use the unchanged nonblocking
group flock: nodes skip groups another node already owns. A stale snapshot
cannot skip a group or override its completed marker; remaining known groups
are appended after prioritized groups. No group identity, shard assignment,
receipt, or frozen source is altered. The priority file contains only
execution hints; it is not durable pipeline state.

### SAM resource lifetime

Reference Edit retains its existing one-SAM-handle-per-GPU executor. When the
drained resource dispatch first enters SA and
`POST_MASK_SA_SAM_PROCESSES_PER_GPU=4`, its eight original SAM predictors
are closed; the SA-only pool then spawns **four model processes per physical
GPU** (32 total) and its CPU/GPU/CPU executor admits at most 96 logical jobs.
A clip is pinned to one SAM child during the stage so the existing physical
session reuse remains effective. Qwen and Boogu remain available for SA
completion and review. With the flag absent (default `1`) the existing
pipeline remains unchanged. `2` is also supported for resource-budget
experiments. This is concurrent independent requests, NOT a semantic
`batch_size=4` inside one SAM predictor.

Prepare and NPY Persist run in parallel thread pools (8 and 8 by default in
4-process mode), using the existing `atomic_write_bytes`, original NPY
layout and original receipt publication. The SAM GPU/process slot is released
once masks are CPU-owned. **No receipt is committed before NPY persistence
succeeds.** Existing terminal-clip binary GC stays asynchronous. There is no
production-wide Triton NMS mutex; the temporary mutex in the earlier 8-GPU
benchmark was only a workaround for concurrent autotuning in one process.

Do not copy a working set of live production RunStorage into a smoke root.
First run focused CPU tests:

```bash
cd "$DEV"
PYTHONPATH="$DEV" "$PY" -m pytest -q \
  tests/test_v3_post_mask_sa_four_process.py \
  tests/test_v3_post_mask_epoch_sa_execution.py \
  tests/test_v3_post_mask_epoch_sa_pipeline.py \
  tests/test_v3_post_mask_full_production.py
```

Then an isolated SA-stage / full-stack smoke with real Qwen + Boogu + 32 SAM
processes **co-resident** on the same eight GPUs, with real SAM masks and
correct independent checkpoint roots, before formal rollout. The
one-GPU SAM-only benchmark does NOT establish co-resident VRAM capacity.
An OOM in an SA child is treated as an epoch-fatal infrastructure error, not
as an accepted `sam_failed` receipt. Measure model startup, SAM jobs/s,
SA terminal clips/s, peak VRAM, masks, persisted receipts, and interruption
resume. Do not assume a 2.4x production SA speedup from the microbenchmark.
