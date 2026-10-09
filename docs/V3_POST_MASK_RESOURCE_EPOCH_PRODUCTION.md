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
`0/1`). They rotate equal-priority ties only. Every node sees every canonical group and
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

The committed config selects Boogu removal and pins all Qwen services to
`Qwen3-VL-8B-Instruct`; the launcher pins the managed physical Qwen model after
loading `server_env.sh`. `ResourceEpochManager` lazily loads and retains Qwen,
Boogu and SAM together for the shared session: sequential work does not imply
exclusive model residency. No Audio/H3 stage is launched here.

## On-demand export of completed groups (operator-run only)

Once a group has the formal
`in_pair_reference/state/resource_epochs/group-XXXXXX/completed.json`,
its published shards can be converted to final
`r2v.v3.production_sample.1` JSONL without waiting for all 48 groups or
running full-population compaction. The independent converter uses the published
shard `samples.jsonl` and `enriched_samples.jsonl` (including SA), and
checks formal group and per-shard completion markers/sample counts. Outputs
are under
`/mnt/workspace/litengjie/data/r2v_v3_group_exports/post_mask/<group>/samples.jsonl`.
Image reference paths are relative to the group directory. A small set of
directory symlinks points to already published shard Export `references`;
**no media is copied**. Keep those source reference directories available.
This does not modify production worktrees, state, receipts or shard exports.

`tools/export_completed_postmask_group.py` lives in the independent
completed-group exporter branch, **not** this SA branch. Fetch its previously
successful pinned version only if missing, into a private tools path
(no changes to a live production checkout):

```bash
D=/mnt/workspace/litengjie/data
PROD="$D/R2V_DATA_V2_postmask_prod"
PY="$PROD/.venv/bin/python"
OUT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/in_pair_reference
RUNS="$D/r2v_v3_runs/production/jea_motion_v1/in_pair_reference"
DEST="$D/r2v_v3_group_exports/post_mask"
SCRIPT="$D/export_completed_postmask_group.py"

if [ ! -f "$SCRIPT" ]; then
  curl -fL \
    "https://raw.githubusercontent.com/TjieLee/R2V_DATA_V2/7e595596916c8c5b517c2d8ec814c6c40977bf38/tools/export_completed_postmask_group.py" \
    -o "$SCRIPT"
fi
ls -lh "$SCRIPT"
```

**List newly completed groups**, leaving prior exports alone:

```bash
for marker in "$OUT"/state/resource_epochs/group-*/completed.json; do
  [ -f "$marker" ] || continue
  g=$(basename "$(dirname "$marker")")
  if [ -f "$DEST/$g/samples.jsonl" ]; then
    echo "$g  ALREADY EXPORTED"
  else
    echo "$g  NEW - READY TO EXPORT"
  fi
done
```

**Batch export all new completed groups**, stopping on the first error.
An existing output directory or symlink is never overwritten:

```bash
for marker in "$OUT"/state/resource_epochs/group-*/completed.json; do
  [ -f "$marker" ] || continue
  g=$(basename "$(dirname "$marker")")
  target="$DEST/$g"
  if [ -e "$target" ] || [ -L "$target" ]; then
    echo "SKIP existing: $g"
    continue
  fi
  echo "========== EXPORT $g =========="
  "$PY" "$SCRIPT" \
    --group "$g" \
    --repo "$PROD" \
    --root "$OUT" \
    --runs "$RUNS" \
    --output-dir "$target" || break
done
```

To export **one** specific completed group instead, use the same command with
`--group group-000001` (replace with an actually completed group) and
`--output-dir "$DEST/group-000001"`.

**Review final JSONL record counts**:

```bash
for f in "$DEST"/group-*/samples.jsonl; do
  [ -f "$f" ] || continue
  echo "$(basename "$(dirname "$f")"): $(wc -l < "$f") samples"
done
```

An SA-completed group without the formal post-Export `completed.json` marker
is **not** eligible. These shell loops only inspect group markers and
existing group JSONLs; they do not scan SAM binaries or source images.
Each JSONL's reference paths remain valid only while its group-relative
symlinks and the published shard image directories exist.

## Group ordering and SA-only process opt-in

Completed groups still skip using formal markers; existing group/shard locks
remain authoritative. Unfinished groups prioritize SA-completed awaiting Export,
then SA-started (descending terminal/eligible ratio when hints exist), Instruct,
Reference Integrity, Reference Edit, Pair, earlier started groups, then fresh
groups. Default ordering reads only bounded group/stage markers. Optional hints
are advisory, never checkpoint/identity: absent/invalid hints fall back to marker
ordering. Generate them once while all nodes are stopped, using the reviewed
checkout and a private output outside state:

```bash
SA_REVIEWED_REPO=${SA_REVIEWED_REPO:?Set the reviewed independent checkout}
SA_HINT_STATE=${SA_HINT_STATE:?Set the campaign state root to read}
SA_HINT_FILE=/mnt/workspace/litengjie/data/sa-group-priority-hints.json
/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python \
  "$SA_REVIEWED_REPO/tools/create_v3_post_mask_priority_hints.py" \
  --state-root "$SA_HINT_STATE" --output "$SA_HINT_FILE" || exit 1
```

The helper reads SA-started eligible UID scopes and counts corresponding direct
outcome-directory entries via `os.scandir`; it does not open each outcome or
scan the whole artifact tree. All ranks may read the same file through the
formal launcher's `--priority-hints` option. The generic smoke launcher below
does not accept that option.

Verified forwarding is the existing `scripts/run_v3_post_mask_full_production.sh`
(`"$@"`) → `tools/run_v3_post_mask_full_production.py` →
`run_elastic_groups(priority_hints=...)`. The external
`run_v3_post_mask_cluster_dynamic.sh` was not found in this checkout or the local
workspace/`/tmp` lookup; its forwarding cannot be verified here. No replacement
launcher is introduced. Before a cluster restart, confirm that external wrapper
passes trailing arguments unchanged, or have the existing job launch the formal
wrapper directly on **every node** as shown below. Merely passing hints to a
wrapper that discards them does not enable ratio ordering.

`POST_MASK_SA_SAM_WORKERS_PER_GPU` accepts only `1`, `2`, `4`; unset means `1`.
On eight GPUs these mean respectively 8/16/32 SAM workers and 24/48/96 bounded
SA admissions. Default `1` retains the existing handles/API. Opt-in `2`/`4`
uses independent spawned SAM processes only during SA; CPU prepare/persist
remain two workers each. Parent SAM handles close before the child pool loads
(no extra fifth SAM model per GPU), and returning to Reference Edit restores
the original worker API. Qwen/Boogu residency and model settings are unchanged.
Validate three-model co-resident VRAM, not SAM-only VRAM.

In spawn mode, only explicit missing-frame/empty-prompt checks before inference
retain clip-local `sam_failed` handling. Backend exceptions (including Triton
`TypeError`), OOM, IPC failures and child loss remain infrastructure failures:
the existing scheduler leaves those jobs retryable/uncommitted, while durable
successful siblings are retained. Children return CPU-owned masks; only the
existing two parent persist workers write NPY/fsync before successful receipts.

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
Boogu/SAM workers (one per physical GPU 0–7); all three models can remain resident.

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
  repo="$1"; tag="$2"; mode="$3"; sam_workers="${4:-1}"
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
    POST_MASK_SA_SAM_WORKERS_PER_GPU="$sam_workers" \
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
run_sa_smoke "$SA_NEW_REPO" "${SA_NEW_TAG}-w1" dry 1 || exit 1
run_sa_smoke "$SA_NEW_REPO" "${SA_NEW_TAG}-w2" dry 2 || exit 1
run_sa_smoke "$SA_NEW_REPO" "${SA_NEW_TAG}-w4" dry 4 || exit 1
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
run_sa_smoke "$SA_NEW_REPO" "${SA_NEW_TAG}-w1" run 1 || exit 1
run_sa_smoke "$SA_NEW_REPO" "${SA_NEW_TAG}-w2" run 2 || exit 1
run_sa_smoke "$SA_NEW_REPO" "${SA_NEW_TAG}-w4" run 4 || exit 1
# Resume: same checkout, config, fixture, tag and roots, without overwrite/delete.
run_sa_smoke "$SA_NEW_REPO" "${SA_NEW_TAG}-w4" run 4 || exit 1
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

The 30–100-clip fixture establishes correctness only. Performance comparison
requires an independent single-node 8×H200 run using the same real 256-SAM-job
set for 8×1/8×2/8×4 and a longer accepted
isolated workload spanning multiple 128-clip SA working-set batches. Reuse the
function with that existing fixture and fresh tags/roots, keeping accepted
models/config and `POST_MASK_CPU_WORKERS=32` fixed. First record co-resident
VRAM, then SAM jobs/s, SA loaded-wave clips/s, GPU utilization, peak host RAM,
persist queue wait, failure counts and output completeness. Do not replace
these with whole-pipeline timing. Small pixel differences are accepted; missing
masks, abnormal attribute counts, model failures or duplicated durable work are
not. Add no hashes, seed checks or full-image audit.

User-supplied single-H200 SAM-only measurements for 256 jobs were
1.587/2.621/3.819 jobs/s at 1/2/4 processes. These are not independently verified
here and do not establish full-SA or three-model co-resident throughput.

Stop after reviewing the smoke and resume results. This procedure does not
authorize deployment, merge, production restart or changes to formal outputs.

## Deployment / rollback instructions (not executed)

Only after review and successful real 8-GPU SA smoke: stop **all** old campaign
processes first, switch the shared production worktree to the reviewed immutable
version once, and confirm its SHA on every node before restart. Never hot-update
a running shared checkout. Preserve existing `$OUT/state`, private RunStorage
roots, receipts and exports unchanged; do not migrate/delete state. PyTorchJob
must inject each node's real `RANK`/`WORLD_SIZE`; do not substitute `0/1`:

```bash
# Post-approval instruction only; run through the existing node/job environment.
: "${RANK:?PyTorchJob must supply the real node rank}"
: "${WORLD_SIZE:?PyTorchJob must supply the real node count}"
POST_MASK_SA_SAM_WORKERS_PER_GPU=4 \
  bash /mnt/workspace/litengjie/data/R2V_DATA_V2_postmask_prod/scripts/run_v3_post_mask_full_production.sh \
  --priority-hints /mnt/workspace/litengjie/data/sa-group-priority-hints.json
```

Run that same argument list on all nodes with their own injected rank and the
same hint file. For the isolated 1/2/4 smoke above, keep two CPU prepare workers,
two persist workers and the accepted models/config unchanged; real co-resident
VRAM and full-SA throughput remain unverified until the operator runs it.

If 4 workers/GPU OOM under Qwen/Boogu/SAM co-residency, stop all processes,
report actual VRAM/failure evidence, and evaluate `2` in isolation; do not swallow
OOM or force deployment. Set the environment option to `1` to restore default
SA execution after stopping processes. A code rollback likewise stops all nodes
first, restores the prior reviewed version once, and resumes the same roots
through the existing wrapper and real rank assignment.
