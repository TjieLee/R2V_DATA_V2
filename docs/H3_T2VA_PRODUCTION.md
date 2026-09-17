# Target-Side T2VA / TA2VA Production

This orchestration is independent of Visual and writes only its owned production
root. It calls the frozen T2VA v7/backend .7/core .6 and TA2VA .2/profile v2
implementations; the reused audio finalizer remains v6. The standalone downstream
runner does not preprocess Audio. The optional full launcher below adds upstream
stage orchestration without migrating legacy outputs or changing eligibility.

## Full Raw-Video Production

Use scripts/run_h3_t2va_full_production.sh for the production path from original
JEA video through finalized T2VA/TA2VA. Production shard size is 2,000 source
rows. One node owns one shard at a time through these barriers:

    canonical -> SAM music_first -> AuK -> resolve -> DiariZen -> ASR -> MiMo

Automatic multi-node scheduling shuffles shard IDs with SHARD_SEED and takes
ordered_shards[RANK::WORLD_SIZE]. RANK/WORLD_SIZE are node-level scheduling
coordinates: different nodes receive different shard sequences. Inside a node,
the eight GPUs do not take eight different shards. They cooperatively process
the current shard: persistent stage workers partition pending jobs across GPUs,
while MiMo uses the full eight-GPU serving topology.

MiMo starts once per launcher invocation and stays alive across stages/shards.
The accepted full-production launcher uses TP=8, DP=2 and
--mem-fraction-static 0.50; the previous 0.65 setting OOMed AuK when all resident
upstream workers were present. Client REQUEST_WORKERS defaults to 1. Upstream
stages use GPU_IDS=0,1,2,3,4,5,6,7 by default. SAM, AuK, DiariZen and Qwen3-ASR
each have eight node-lifetime resident workers (32 upstream workers, in addition
to MiMo). Every worker loads once before its READY handshake, processes requests
across shards and closes only at node shutdown. Stage barriers remain sequential;
residency does not mean concurrent GPU stages. Each child sees its physical GPU
through CUDA_VISIBLE_DEVICES and uses cuda:0; the parent environment is unchanged.
There is no automatic GPU memory manager or model restart daemon.

Canonical preparation uses CANONICAL_WORKERS=16 (Python --canonical-workers),
configurable to any positive integer. Per-clip publication and hash checks are
unchanged; the final manifest is always in source order. An owned CPU subprocess
prepares at most one next shard while the current shard runs its GPU stages.
First-shard canonical preparation starts before pool initialization, so CPU work
can overlap model startup. Every canonical path takes blocking canonical.lock;
invocation.lock still prevents duplicate GPU-stage writers. Background canonical
also waits for that lock before canonical.lock, so another node cannot republish
the same shard's input manifest during active GPU consumption. Termination cleans
the prefetch process and its FFmpeg children. Prepared state is per shard, not a
mutable current Audio root.

Within each ASR worker, a single CPU thread prefetches the next waveform.
Transcription remains serial (one GPU inference at a time). Loader failures are
attributed to their own segment. SAM/AuK postprocessing is not pipelined.

The launcher sources the repo .venv and server_env.sh. server_env.sh remains the
local authority for the SAM runtime settings. The full launcher supplies the
standard production defaults for AuK, DiariZen, Qwen3-ASR, SGLang and MiMo paths,
while still allowing explicit environment overrides. No dependency is downloaded.
The launcher does not edit server_env.sh. It adds localhost to NO_PROXY/no_proxy
without removing external proxies. Readiness checks owned PID, /v1/models and
/model_info.

### Cluster launch: current production form

The formal production root is:

    /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA

Use the existing cluster job wrapper. Do not hard-code a node count into the
wrapper. The scheduler supplies a unique RANK and a common WORLD_SIZE for each
allocation, so the same file works when a resumed allocation has a different
number of nodes. Keep the fallbacks only for single-node/manual use:

```bash
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

export PRODUCTION_ROOT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA
export SHARD_SEED=20260918
export ALLOW_UNVERIFIED=1

cd /mnt/workspace/litengjie/data/R2V_DATA_V2
bash scripts/run_h3_t2va_full_production.sh
```

Each allocated node is expected to have eight GPUs. The full launcher already
defaults GPU_IDS=0,1,2,3,4,5,6,7, CANONICAL_WORKERS=16 and REQUEST_WORKERS=1,
so the cluster wrapper should not repeat those values unless deliberately
overriding them. Likewise, do not copy the long AuK/DiariZen/ASR path export list
into the cluster wrapper; the launcher owns those standard defaults.

All nodes in one allocation must see the same shared PRODUCTION_ROOT, source
manifest, clips root and source-video root, and must receive unique RANK values
within the same WORLD_SIZE. Keep SHARD_SEED=20260918 fixed for this production
population so scheduling is reproducible. The number of nodes may change after an
interruption: restart with the new scheduler-provided WORLD_SIZE and the same
production root/seed. The new rank slicing still partitions the full shard list;
already-ready receipts are reused, and shard/invocation locks protect the shared
root. Do not change shard size or source identity in-place.

For manual debugging only, explicit SHARDS bypasses rank/world allocation:

```bash
SHARDS=59,12,87 ALLOW_UNVERIFIED=1 \
  bash scripts/run_h3_t2va_full_production.sh
```

### Cluster filesystem / Python runtime prerequisite

/mnt/workspace is shared, but a venv executable may still be a symlink to a
node-local path. This caused the cluster launcher to fail immediately with:

    Missing AUK_PYTHON: /mnt/workspace/litengjie/data/audio_deps/auk/auk-venv/bin/python

The AuK venv path itself existed on shared storage, but bin/python originally
resolved through /opt/uv/python/... on the notebook. Ephemeral compute nodes did
not provide that notebook-local /opt/uv target. Do not try to repair /opt/uv on
each temporary node. Make the venv resolve to a shared runtime (or to a Python
provided by the common base image) once on shared storage.

The accepted AuK layout resolves to the shared runtime alias:

    /mnt/workspace/litengjie/data/shared_uv_runtime/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10

DiariZen already uses a shared audio_deps/uv-python runtime. Qwen3-ASR and the
SGLang environment currently resolve to /usr/bin/python3.12 and therefore rely on
the common cluster base image providing that interpreter.

Before changing or recreating any environment, inspect resolution without exiting
the current shell:

```bash
for py in \
  /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python \
  /mnt/workspace/litengjie/data/audio_deps/auk/auk-venv/bin/python \
  /mnt/workspace/litengjie/data/audio_deps/diarizen-venv/bin/python \
  /mnt/workspace/litengjie/data/audio_deps/qwen3-asr-venv/bin/python \
  /mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env/bin/python
do
  echo "===== $py ====="
  readlink "$py" 2>/dev/null || true
  readlink -f "$py" 2>/dev/null || true
done
```

A shared path that ultimately resolves back to notebook-local /opt/uv is not
cluster-portable even though ls shows the venv under /mnt/workspace. Prefer
shared, versioned Python runtime directories with relative aliases; avoid
absolute aliases back into /opt/uv.

Do not supply an existing Audio cache to the full runner. It owns
shards/<shard>/audio_production, including canonical_items, audio manifests and
sam_audio_stem_shadow_v1/runs/production/{separation,auk_speech_v1,
resolved_stems_v1,diarization,asr}. The JEA bootstrap has no Visual references or
binding sidecars. It uses the same default target-side loader path as the frozen
T2VA fixtures, not the RA2VA binding_evidence_mode=none Visual adapter. The name
of that default mode does not introduce LR-ASD execution or evidence.

Each stage has durable job receipts beneath stage_state/<stage>/jobs.
Persistent worker control uses owned request files, never model stdout.
Per-GPU logs append beneath workers/<stage>-gpu<ID>.log; stage summaries and
exceptions remain under shard logs. Canonical subprocess logs are
logs/canonical-prefetch.log within the shard. Pool startup and stage wall seconds,
plus canonical prefetch start/completion, are printed in the supervisor log.
Node logs are logs/<hostname>/{mimo,supervisor}-<timestamp>-<pid>.log.
Only the parent publishes full ordered inventories. Worker outputs never race on
canonical records.jsonl. Successful receipt/media hashes are checked before
reuse; failed jobs retry once on the next invocation. A worker crash waits for
other workers, stops the supervisor, and leaves unattempted jobs pending.
Ctrl-C/TERM terminates owned worker trees. Never delete ready receipts merely to
rerun the stage.

stage_state/<stage>/invocation.json records scheduled_job_count,
reused_ready_count and worker_count for the latest invocation. Zero scheduled
jobs means no inference request is sent; already-resident workers remain idle.
Frozen publication model counts still describe the cached artifacts, not new
calls in this invocation. Keep interpreters and dependency environments pinned
for a resumed run; use a fresh production root when changing source identity,
shard size or incompatible runtime semantics rather than adopting old caches.

Upstream aggregate manifests can expand after failed clips recover. The full
downstream adapter binds ready outputs to validated per-clip dependencies;
unrelated aggregate hash changes cannot trigger repeat inference. Previously
upstream-skipped clips can reopen when their own inputs become available.
Published data remains available to the existing snapshot builder.

### Raw-video server acceptance (2026-09-18)

The performance-v2 raw-5 server run passed with the resident upstream pools and
MiMo --mem-fraction-static 0.50. The final stage summary was:

    canonical  ready=5 failed=0
    SAM        records=5 model_call_count=10
    AuK        ready=5 failed=0 model_call_count=5
    resolve    ready=5 failed=0
    DiariZen   jobs=5 ready=5 failed=0
    ASR        jobs=6 ready clips=5 failed=0
    MiMo       t2va_ready=5 ta2va_ready=5, all failed/pending/skipped=0

ASR job_count can exceed ready clip count because ASR jobs are speech segments.
The accepted run exited with no owned worker processes left behind. A prior run
with MiMo mem-fraction-static 0.65 OOMed AuK during inference because MiMo held
about 92 GiB per GPU while the resident audio models were also present; 0.50 is
the current production setting. Full-population cluster production was started
only after this acceptance and the shared-Python runtime issue above was fixed.

For future runtime/architecture changes, repeat a small raw-video smoke before
resuming large production. Use a fresh smoke root for changes that alter source
index identity or incompatible runtime semantics; do not reuse the historical
10k-shard smoke root.

## Standalone Downstream Prerequisites

The population authority is the original ordered JEA shot JSONL, not an Audio
or Visual inventory. Production shard size is 2,000. Each physical source row
belongs permanently to a 2,000-row shard, including malformed/unavailable rows.
Shard 59 is source indexes 118000 through 119999. Shards 0 and 1 cover 0..1999
and 2000..3999. The last shard can be shorter; shard count is derived from the
actual source.

An old 10,000-row-shard root is incompatible and raises
"source index shard size mismatch". Use a fresh root, including for raw-5 smoke;
do not reuse raw5-20260918-005405/production. No old root is rewritten,
deleted or migrated. Shard size is part of source-index identity.

The selected clips need matching, finalized canonical Audio and a named
resolved AuK/SAM -> DiariZen -> Qwen3-ASR cache. Missing upstream data retains the
existing skipped semantics; this runner does not launch those models. Supply an
Audio root/run covering the intended production population. Do not point a
full-population run at a random20-only cache and expect Audio to be generated.
Skipped clips are not automatically retried; use a new production root when
changing the source/config/upstream population.

The existing tools/export_h3_training_manifests.py exports completed shadow
products. The production snapshot writer independently supports incremental
publication using the same four loader-facing fields. There is no independent
target-side T2V producer: the existing Visual t2v_caption field is not a separate
no-reference target-side contract. No new T2V dataset or call is invented.

## Preflight

Default destination (explicitly authorized for this task):

    /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA

The runner creates this directory and checks writability; it never silently
chooses another root. Source datasets and model files remain read-only.
Index/cache writes are under the output root, not beside the input JSONL.

Set existing server-local paths, then inspect without starting any model:

```bash
export AUDIO_PRODUCTION_ROOT=/path/to/finalized/target-side-audio
export AUDIO_SHADOW_RUN_ID=resolved-run-id
export SHOT_MANIFEST=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl
export JEA_CLIPS_ROOT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped
export JEA_SOURCE_VIDEOS_ROOT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808
RANK=0 WORLD_SIZE=4 SHARD_SEED=20260917 \
  bash scripts/run_h3_t2va_production.sh --dry-run
```

Shell dry-run prints the exact commands without sourcing local environments or
writing files. Python --dry-run builds/reuses the source index and reports
schedule/count, but neither constructs a model client nor calls inference:

```bash
python tools/run_h3_t2va_production.py \
  --shot-manifest "$SHOT_MANIFEST" --clips-root "$JEA_CLIPS_ROOT" \
  --source-videos-root "$JEA_SOURCE_VIDEOS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --audio-shadow-run-id "$AUDIO_SHADOW_RUN_ID" \
  --media-root /mnt/workspace --rank 0 --world-size 4 \
  --shard-start 0 --shard-end 379 --shard-seed 20260917 --dry-run
```

Do not specify shard 379 unless the manifest actually has that shard. Without
--shard-end the runner uses the actual last shard.

## Launch

```bash
RANK=0 WORLD_SIZE=4 SHARD_SEED=20260917 \
  bash scripts/run_h3_t2va_production.sh

# Alternatively: explicit execution order, bypassing rank/world allocation.
SHARDS=59,12,87 bash scripts/run_h3_t2va_production.sh
```

Every host uses the same shared production root and source manifest. Automatic
scheduling shuffles only shard IDs and takes ordered_shards[rank::world_size].
One nonblocking shard lock prevents duplicate writers; a second invocation lock
covers preparation. The shared source-index build waits for its first writer.
No cross-machine work stealing or database is used.

The launcher sources the existing .venv and server_env.sh, clears PYTHONPATH,
starts the exact accepted SGLang command from H3_AUDIO_SERVER_RUNBOOK.md on
127.0.0.1:8092, waits for /v1/models, and runs Python. It refuses to adopt an
already-listening endpoint and only terminates its own MiMo/runner PIDs.
Logs are under logs/<hostname>/{mimo,production}-<timestamp>-<pid>.log.

REQUEST_WORKERS defaults to 8. ALLOW_UNVERIFIED=1 passes the existing explicit
pilot permission for unverified resolved stems; it is not silently enabled.
All additional shell arguments are forwarded to the production Python runner.
For an already managed endpoint, call the Python runner directly.

## Durable Publication and Resume

Each shard has:

- state.jsonl.partial: fsynced append-only latest stage transitions;
- sources.json: internal source index/clip/video mapping, not training data;
- prepared/: frozen inventory adapter provenance;
- artifacts/<clip_uid>/{t2va,ta2va}/: complete atomic stage directories;
- exports/{t2va,ta2va_full_audio,ta2va_speech_bgm}.jsonl.partial;
- COMPLETE once all rows are ready or upstream-skipped.

An unavailable or malformed sample does not stop its neighbors. A T2VA failure
skips TA2VA; a TA2VA failure retains the T2VA artifact and export. Each failed stage
gets one attempt per invocation. Repeat the same command to retry failed stages.
Already ready T2VA is not called again when only TA2VA failed. An ineligible
speech track yields a successful full-audio-only TA2VA stage.

If speech profiling fails, the independently valid full-audio variant is still
published. Its receipt/diagnostics live in ta2va_failed/ until a later successful
TA stage is available. The TA status remains failed and only that stage is retried.
This preserves the frozen runner's per-variant availability semantics.

Stages publish a complete directory before appending ready state. A crash in
between is recovered from the directory's completion envelope and file hashes.
The source/config identity must remain unchanged for resume. Orphan temporary
directories are never considered ready. Export lines are written in full,
flushed and fsynced. A torn final line is repaired by its shard writer before
appending; readers ignore non-newline-terminated tails.

Connection-refused/API connection failures stop scheduling and exit nonzero.
At most the already active workers can still complete; unscheduled clips are
not labeled failed. Restart the endpoint/launcher to resume. There is no model
retry loop or automatic model restart daemon.

## Incremental Snapshots

```bash
python tools/build_h3_t2va_snapshot.py \
  --production-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA \
  --snapshot-id 20260917-200000
```

No shard must be complete: 327 published rows in an incomplete shard are already
exportable. The builder consumes complete rows from both final and partial files,
deduplicates by clip/task, and fails on conflicting content rather than guessing.
Caption memory is bounded to one shard; global duplicate tracking stores hashes.

Snapshots are immutable, atomically published directories containing videos.jsonl,
t2va.jsonl, ta2va_full_audio.jsonl, ta2va_speech_bgm.jsonl and summary.json.
LATEST is an atomically replaced plain text file, not a symlink.

Training rows have exactly:

```json
{"video": "/path/video.mp4", "images": [], "audios": [], "caption": "..."}
```

TA2VA audios contains its actual conditioning file paths. Internal hashes,
identities, schemas and statuses are never added to these rows. videos.jsonl
contains only video and the currently available task list. Future snapshots
gain later TA2VA products without modifying earlier snapshots.

## Acceptance Before Semantic Changes

CPU tests cover frozen-vs-production core/prompt equivalence, TA2VA PCM hashes,
resume without repeat calls, concurrent writes, interruption recovery,
incremental snapshots and launcher lifecycle with synthetic executables. The
full raw-video path additionally has the real raw-5 server acceptance documented
above.

If frozen T2VA/TA2VA semantics, upstream model versions, shard identity or
serving/runtime architecture changes, create a separate smoke output root and
repeat a small real endpoint validation before resuming large production. Compare:

1. production artifacts/<uid>/t2va/core.json to frozen core semantic payload;
2. T2VA prompt and exact dialogue;
3. TA2VA variant set, six-section prompts and reuse PCM hashes.

Do not treat a CPU-only test pass as evidence for a changed real-model runtime.
For ordinary cluster interruptions with unchanged code/config/source identity,
resume the same shared production root and let durable receipts/locks recover the
remaining work.