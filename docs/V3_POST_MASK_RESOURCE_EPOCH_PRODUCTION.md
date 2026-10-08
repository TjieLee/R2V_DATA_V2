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
