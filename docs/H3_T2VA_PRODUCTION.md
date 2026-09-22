# Target-Side T2VA / TA2VA Production

This orchestration is independent of Visual and writes only its owned production
root. It calls the frozen T2VA v7/backend .7/core .6 and TA2VA .2/profile v2
implementations; the reused audio finalizer remains v6. The standalone downstream
runner does not preprocess Audio. The optional full launcher below adds upstream
stage orchestration without migrating legacy outputs or changing eligibility.

## MiMo hot-swap policy inside one production root

The production root and all T2VA/TA2VA shard paths are independent of the MLLM
version. Changing MiMo must not create a new downstream directory, invalidate
existing ready artifacts, or require a fresh production root.

Current production default remains MiMo V2.5 until the fixed random20 V2.6
multi-call A/B passes:

    model: mimo-v2.5
    checkpoint: /mnt/workspace/public/pretrained/MiMo/MiMo-V2.5

V2.6 candidate:

    model: mimo-v2.6-flash-rl
    checkpoint: /mnt/workspace/public/pretrained/MiMo/MiMo-V2.6-Flash-RL

When the served model is `mimo-v2.6-flash-rl`, the full-production launcher
automatically enables the official EAGLE/MTP speculative decoder parameters.
There is no separate speculative on/off operator switch.

Existing V2.5-ready artifacts remain valid at their original
`<PRODUCTION_ROOT>/shards/.../artifacts` paths. Pending/failed samples continue
in the same shard/state/export layout using the currently configured MiMo model.
The model name is runtime observability/provenance only; it is not a path or
resume-cache key in full production.

Use `MIMO_MODEL` to hot-swap the runtime model while keeping the same
`PRODUCTION_ROOT` and data organization.

MiMo checkpoint contents are not part of the production-root identity and are
not content-hashed by the T2VA/TA2VA serving path. Full-production DiariZen and
Qwen3-ASR runtime provenance also uses lightweight logical model identity rather
than rereading whole model directories. Legacy SAM/AuK schemas still contain
historical checkpoint/dependency SHA fields so existing upstream receipts remain
cache-compatible; removing those fields requires a separate compatibility
migration and is intentionally not coupled to an MLLM upgrade. Source media and
generated artifacts keep their content hashes.

MiMo V2.6 uses the same OpenAI-compatible multimodal request format. The existing
multi-call prompts/turns stay unchanged, while V2.6 always uses the official
EAGLE/MTP speculative serving configuration for best runtime performance.

## Full Raw-Video Production

Use scripts/run_h3_t2va_full_production.sh for the production path from original
JEA video through finalized T2VA/TA2VA. Production shard size is 2,000 source
rows. One node owns one shard at a time through these barriers:

    canonical -> SAM music_first -> AuK -> resolve -> DiariZen -> ASR -> MiMo

Automatic multi-node scheduling first shuffles shard IDs with SHARD_SEED and
takes ordered_shards[RANK::WORLD_SIZE], fixing cross-node ownership before any
resume inspection. Each node then reorders only its own assigned shards:
unfinished shards first, never-started shards second, and shards with COMPLETE
are omitted from automatic scheduling. Resume classification is intentionally
metadata-only and batched: it enumerates shards/ once, treats any existing shard
directory without a top-level COMPLETE file as unfinished, and treats absent
shard directories as fresh. It never opens stage JSON, receipts, media, or hashes
during scheduling. Explicit SHARDS preserves the caller's exact order and
bypasses this automatic resume-first filtering.

RANK/WORLD_SIZE are node-level scheduling coordinates: different nodes receive
different shard sequences. Inside a node, the eight GPUs do not take eight
different shards. They cooperatively process the current shard: the active
upstream stage partitions pending jobs across GPUs, while the downstream MiMo
stage uses the full eight-GPU serving topology.

No large GPU model is node-lifetime resident. SAM, AuK, DiariZen and Qwen3-ASR
use the existing ephemeral executor: each stage starts at most one worker per
physical GPU, waits for its barrier, then terminates those workers before the next
GPU stage. MiMo is also shard-local and lazy. Entering the downstream stage does
not load MiMo; the first actual T2VA/TA2VA API request starts SGLang with TP=8,
DP=2 and --mem-fraction-static 0.65, and leaving that shard's downstream stage
stops it. A fully cached downstream shard therefore never starts MiMo. This keeps
MiMo completely absent while SAM/AuK/DiariZen/ASR run and prevents their memory
peaks from overlapping.

Client REQUEST_WORKERS defaults to 1 and GPU_IDS defaults to
0,1,2,3,4,5,6,7. Each upstream child sees its physical GPU through
CUDA_VISIBLE_DEVICES and uses cuda:0; the parent environment is unchanged.
There is no automatic model retry or restart daemon within a shard.

Canonical preparation uses CANONICAL_WORKERS=16 (Python --canonical-workers),
configurable to any positive integer. Per-clip publication and hash checks are
unchanged; the final manifest is always in source order. An owned CPU subprocess
prepares at most one next shard while the current shard runs its GPU stages.
Every canonical path takes blocking canonical.lock;
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
without removing external proxies. The shell launches only the Python supervisor.
For each shard that needs a real downstream request, the Python MiMo lifecycle
checks that 127.0.0.1:8092 is free, starts its owned SGLang child, waits for both
/v1/models and /model_info, and tears the server down at the downstream barrier.

### Cluster launch: current production form

The formal production root is:

    /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA

Use the dedicated production worktree, not the mutable development checkout.
The development checkout at /mnt/workspace/litengjie/data/R2V_DATA_V2 may switch
branches freely; cluster production must always launch from:

    /mnt/workspace/litengjie/data/R2V_DATA_V2_h3_prod

Keep that worktree on the Audio/H3 production line and update it explicitly
before a new cluster submission:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2_h3_prod

git fetch origin feature/h3-audio-jea-qwen3-v1
git merge --ff-only origin/feature/h3-audio-jea-qwen3-v1

git rev-parse HEAD
```

Do not use the mutable main checkout as the cluster launcher path. A previous
failure occurred after the main checkout switched branches and no longer contained
tools/run_h3_t2va_full_production.py. The full launcher now passes the runner by
absolute REPO_ROOT path, but the dedicated worktree remains the production
authority.

Do not hard-code a node count into the wrapper. The scheduler supplies a unique
RANK and a common WORLD_SIZE for each allocation. Keep fallbacks only for
single-node/manual use. The current cluster wrapper can remain minimal:

```bash
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29506}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

# Required by OmniShotCut-related runtime code used elsewhere in this environment.
export TORCH_HOME=/mnt/workspace/public/.cache/

export PRODUCTION_ROOT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA
export SHARD_SEED=20260918
export ALLOW_UNVERIFIED=1

echo "hostname=$(hostname) rank=$RANK world_size=$WORLD_SIZE"

exec /bin/bash \
  /mnt/workspace/litengjie/data/R2V_DATA_V2_h3_prod/scripts/run_h3_t2va_full_production.sh
```

The cluster submission itself stays a single command:

```bash
bash /mnt/workspace/litengjie/data/parallel_scripts/run_h3_t2va_cluster_dynamic.sh
```

Do not put `export PRODUCTION_ROOT=...` after an inline `#` comment on the
TORCH_HOME line; shell comments consume the remainder of that line.

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
production root/seed. A changed WORLD_SIZE can move shard ownership to a
different rank, but the new owner still prioritizes that unfinished shard within
its own fixed slice. Already-ready receipts are reused, and shard/invocation
locks protect the shared root. Do not change shard size or source identity
in-place.

For manual debugging only, explicit SHARDS bypasses rank/world allocation:

```bash
SHARDS=59,12,87 ALLOW_UNVERIFIED=1 \
  bash scripts/run_h3_t2va_full_production.sh
```

### Production worktree and Python/runtime prerequisites

The production worktree may reuse the repository venv and server_env.sh through
symlinks, but code and Python runtime ownership are separate concerns. A typical
one-time worktree setup is:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
git fetch origin
git worktree add \
  -b h3-audio-prod \
  /mnt/workspace/litengjie/data/R2V_DATA_V2_h3_prod \
  origin/feature/h3-audio-jea-qwen3-v1

ln -s \
  /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv \
  /mnt/workspace/litengjie/data/R2V_DATA_V2_h3_prod/.venv

ln -s \
  /mnt/workspace/litengjie/data/R2V_DATA_V2/server_env.sh \
  /mnt/workspace/litengjie/data/R2V_DATA_V2_h3_prod/server_env.sh
```

The .venv symlink is acceptable because the launcher now executes
`$REPO_ROOT/tools/run_h3_t2va_full_production.py` with an absolute production
worktree path. Do not regress that runner path back to a relative
`tools/run_h3_t2va_full_production.py`; doing so previously made production
depend on whichever branch was checked out in the main repository.

Separately, /mnt/workspace is shared but a venv interpreter may still resolve to
a node-local path. This previously caused:

    Missing AUK_PYTHON: /mnt/workspace/litengjie/data/audio_deps/auk/auk-venv/bin/python

The AuK venv itself was on shared storage, but bin/python resolved through
/opt/uv/python/... on the notebook. Ephemeral compute nodes did not necessarily
provide that notebook-local /opt/uv target. The accepted fix is to make the AuK
venv resolve directly to a shared versioned Python runtime rather than recreating
/opt/uv on every compute node:

    /mnt/workspace/litengjie/data/shared_uv_runtime/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10

DiariZen uses a shared audio_deps/uv-python runtime. Qwen3-ASR and SGLang
currently resolve to /usr/bin/python3.12 and therefore rely on the common cluster
base image providing that interpreter.

Before changing or recreating environments, inspect the exact resolution without
commands that terminate the current notebook shell:

```bash
for py in \
  /mnt/workspace/litengjie/data/R2V_DATA_V2_h3_prod/.venv/bin/python \
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

A path under /mnt/workspace is not cluster-portable if its final symlink target
still points into notebook-local /opt/uv. Prefer shared, versioned Python runtime
directories or Python from the common base image. Do not add per-node /opt/uv
repair logic to the production cluster wrapper.

Do not supply an existing Audio cache to the full runner. It owns
shards/<shard>/audio_production, including canonical_items, audio manifests and
sam_audio_stem_shadow_v1/runs/production/{separation,auk_speech_v1,
resolved_stems_v1,diarization,asr}. The JEA bootstrap has no Visual references or
binding sidecars. It uses the same default target-side loader path as the frozen
T2VA fixtures, not the RA2VA binding_evidence_mode=none Visual adapter. The name
of that default mode does not introduce LR-ASD execution or evidence.

Each stage has durable job receipts beneath stage_state/<stage>/jobs.
Stage-local worker control uses owned request files, never model stdout.
Per-GPU upstream logs are written beneath the shard logs for the active stage;
stage summaries and exceptions remain under shard logs. Canonical subprocess logs
are logs/canonical-prefetch.log within the shard. Lazy MiMo server logs are
logs/<hostname>/mimo-shard-<shard>-<timestamp>-<supervisor-pid>.log. Stage wall
seconds, canonical prefetch start/completion, and MiMo server start/ready/stop
events are printed in the supervisor log.
The node-level shell log is logs/<hostname>/supervisor-<timestamp>-<pid>.log;
MiMo logs are shard-local lifecycle logs as described above.
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
for a resumed run; use a fresh production root only when changing source
identity, shard size, or incompatible upstream semantics. Changing the MiMo
model continues in the same downstream shard/artifact layout.

Upstream aggregate manifests can expand after failed clips recover. The full
downstream adapter binds ready outputs to validated per-clip dependencies;
unrelated aggregate hash changes cannot trigger repeat inference. Previously
upstream-skipped clips can reopen when their own inputs become available.
Published data remains available to the existing snapshot builder.

### Pre-submit checklist

Before a new cluster submission, update only the dedicated production worktree:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2_h3_prod

git fetch origin feature/h3-audio-jea-qwen3-v1
git merge --ff-only origin/feature/h3-audio-jea-qwen3-v1

git rev-parse HEAD
git status --short
```

If .venv and server_env.sh are intentionally local symlinks, they may appear as
untracked entries in this worktree. Do not commit them and do not replace them
merely to make git status empty.

For lifecycle/scheduling changes, the focused CPU checks are:

```bash
source .venv/bin/activate

bash -n scripts/run_h3_t2va_full_production.sh

python -m compileall -q \
  r2v_data_v2/h3/t2va_mimo_stage_server.py \
  r2v_data_v2/h3/t2va_full_production.py \
  tools/run_h3_t2va_full_production.py

python -m pytest -q \
  tests/test_h3_t2va_mimo_stage_server.py \
  tests/test_h3_t2va_full_production.py \
  tests/test_h3_t2va_full_overlap.py \
  tests/test_h3_t2va_full_launcher.py \
  tests/test_h3_t2va_production.py::test_cross_mount_source_identity_and_changed_content
```

These tests validate orchestration and resume behavior but do not replace a real
GPU smoke or continued production observation after serving/runtime changes.

### Startup diagnostics and resume scheduling

The full supervisor prints startup phase markers to the supervisor log, not to
the outer cluster terminal. The outer terminal normally stops after printing the
supervisor log path. Inspect the referenced log directly, for example:

```bash
tail -f /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/T2VA/logs/<hostname>/supervisor-<stamp>.log
```

A healthy startup begins with:

    source_index_start
    source_index_ready shards=1920 records=3839096
    resume_scan_start assigned=<rank slice size>
    schedule {"complete_skipped": ..., "fresh": ..., "unfinished": ..., ...}

Interpret stalls by marker. If only source_index_start appears, the source-index
path is the bottleneck. If source_index_ready appears but schedule does not,
resume metadata enumeration is the bottleneck.

Do not reintroduce full-manifest SHA validation on every cluster node. The source
index already stores the authoritative SHA from creation. On resumed cluster
mounts, path + size + mtime_ns are used as the fast cross-mount identity; a full
SHA is only the fallback when those content-facing fields change. device, inode,
and ctime may differ across shared mounts and must not by themselves trigger a
multi-million-row manifest rehash.

Resume-first scheduling is intentionally cheap. After the fixed
SHARD_SEED/rank/world-size partition, each node enumerates shards/ once. Existing
assigned shard directories without a top-level COMPLETE marker are prioritized;
absent directories are fresh; COMPLETE shards are skipped. No stage JSON, receipt,
media file, or artifact hash is read for scheduling.

### Canonical prefetch resume policy

Canonical lookahead is CPU/shared-storage work and can leave all GPUs idle while
the parent waits for the current shard's preparation. Resume must therefore avoid
re-reading immutable media and must never wait indefinitely on another
invocation.

For an existing shard, `source/selection.json` is the durable source-selection
authority. If it matches the current shard manifest and configured roots,
canonical prefetch reuses it directly instead of hashing every target video
again. Existing canonical clip receipts are also reused without re-hashing each
cached full-audio file. For a fresh shard, the target-video SHA is established
once when the selection is first created; the per-clip Audio bootstrap does not
hash that same video a second time.

The prefetch child uses a nonblocking `invocation.lock`. If another allocation
currently owns the shard, that shard is skipped for this invocation and the
lookahead advances to the next assigned shard. Do not change this back to a
blocking invocation lock: a blocking lock can make the parent wait for hours
with all GPUs at 0% while another allocation owns the shard. The narrower
`canonical.lock` may still block briefly while a legitimate canonical writer
finishes.

When diagnosing GPU-idle time before SAM, check:

    shard=<id> canonical_prefetch_started
    shard=<id> canonical_prefetch_completed elapsed_seconds=...

If `canonical_prefetch_started` appears without completion, inspect
`shards/<shard>/logs/canonical-prefetch.log` and whether another allocation is
still active on the same production root.

### Speech-stage model fingerprint policy

DiariZen and Qwen3-ASR configuration provenance includes a fingerprint of the
local model path. Do not recompute that fingerprint once per shard. The model
paths are pinned for one production supervisor invocation, and recursively
hashing a model directory means reading every model file from shared storage
before any GPU worker starts.

The full-production speech runner therefore caches the model fingerprint/config
by the relevant environment configuration (model path, model identifier, dtype,
timeouts, etc.). Repeated shards with identical configuration reuse the cached
fingerprint. If the configuration key changes, the fingerprint is recomputed.

The supervisor prints:

    diarizen_config_start
    diarizen_config_ready
    asr_config_start
    asr_config_ready

A long gap between a *_config_start and *_config_ready marker indicates model
configuration/fingerprint work, not GPU inference.

Within an ASR worker, multiple transcript segments from the same speech stem also
reuse one successful source-audio hash validation instead of hashing the same stem
for every segment.

### Downstream source projection and shared-path metadata

Full production already establishes the accepted shard source population during
canonical preparation in `shards/<shard>/source/selection.json`. Downstream
preparation must reuse that selection instead of resolving and stat'ing every
target video/source-video path again.

A previous implementation reparsed all 2,000 manifest rows during
`prepare_shard()` and called the JEA adapter for every row. Each adapter call
performed multiple `Path.resolve()` / `is_file()` operations on shared
storage. With several nodes entering downstream preparation together, this could
create thousands of metadata RPCs while CPU and GPU utilization both appeared
near zero and the supervisor stopped at:

    shard=<id> downstream_prepare_start

When a cached canonical selection exists, downstream source projection now reads
the shard manifest sequentially only to retain the existing source-row byte hash
and reconstructs clip_uid/video directly from the cached selection. It does not
re-resolve or re-stat every media path.

Current markers are:

    downstream_source_projection_start
    downstream_source_projection_ready rows=...
    downstream_inventory_build_start
    downstream_inventory_build_ready jobs=...

Do not reintroduce per-row `Path.resolve(strict=True)`, `is_file()`, or other
shared-media metadata probes in the downstream resume hot path. Actual pending
media remains validated at its consumption boundary.

### Bulk downstream preparation failure policy

Full-production downstream preparation builds one T2VA inventory for the whole
shard. If that bulk inventory build fails, the invocation must fail immediately
and expose the real exception. Do not silently fall back to rebuilding the full
canonical/resolved/diarization/ASR inventory once per clip.

A previous fallback swallowed any bulk ValueError/TypeError/KeyError/OSError and
then called build_t2va_inventory() separately for every row missing from contexts.
For a 2,000-row shard this could reload and revalidate the same shard-global
upstream state up to 2,000 times, leaving all GPUs idle for hours while the
supervisor showed only:

    shard=<id> downstream_prepare_start

Malformed source rows, missing media, and other row-local ingestion failures are
already isolated before the bulk inventory build. Upstream per-clip failures are
represented in the bulk inventory as job.upstream_failure. Therefore a bulk
inventory exception is treated as a shard/infrastructure/lineage error, not as a
reason to replay the entire inventory once per clip.

Current markers are:

    downstream_inventory_build_start
    downstream_inventory_build_ready jobs=...
    downstream_inventory_build_failed <ExceptionType>: <message>

No per-UID prepared/*.jsonl fallback files should be created by normal
full-production resume.

### Downstream inventory complexity

Downstream preparation must be linear in the shard population plus speech
segments. A previous implementation rebuilt T2VA speech facts with this pattern:

    for each selected clip:
        find the resolved job by scanning all resolved jobs
        sort every diarization segment in the shard
        discard segments belonging to other clips

For a 2,000-row shard with roughly 4,500-6,500 ASR/diarization segments, that
became thousands of repeated whole-segment sorts and left all GPUs idle for
hours after ASR while the supervisor stopped at:

    shard=<id> downstream_prepare_start

The production path now builds dictionaries/sets for resolved jobs and clip
membership once, groups raw diarization segments by clip once, sorts only each
clip's own segment list once, and then performs O(1) lookups while constructing
jobs. The downstream dependency processor likewise indexes raw/ASR segment keys
by clip once rather than scanning the complete segment dictionary for every
clip.

Expected phase markers are now:

    downstream_prepare_start
    downstream_prepare_ready
    downstream_processor_start
    downstream_processor_ready
    downstream_preflight_start
    downstream_preflight_ready

If preparation stalls before downstream_prepare_ready, inspect inventory
construction rather than GPU serving. Do not reintroduce per-clip scans or sorts
over shard-global resolved jobs, raw segments, or ASR segments.

### Downstream hash policy and GPU-idle diagnosis

MiMo is lazy, so CPU/shared-storage work before the first real downstream request
appears as 0% GPU utilization. Do not put repeated full-media SHA validation in
that hot path.

Full production reuses the canonical stage's
`shards/<shard>/source/selection.json` when constructing the T2VA inventory.
That selection already carries the target-video hashes established during
canonical preparation, so downstream preparation must not re-read all 2,000
videos merely to recompute the same hashes. Likewise, resolved/canonical records
already carry the per-clip Audio hashes; inventory construction does not rehash
all full/speech/music/sfx files.

Aggregate `inventory.source_hashes` are validated once per unique full-production
inventory by the full downstream processor. They are not rehashed before and
after every T2VA/TA2VA sample. Standalone/frozen production keeps its default
strong per-sample source verification unless explicitly using the full-production
prevalidated path.

Integrity checks that remain in the hot path are intentionally local:

- each pending full-production T2VA sample validates its target video and four
  Audio inputs once before use; the backend/stage do not repeat the same hashes
  before and after the request;
- TA2VA does not repeat a whole-media start/end _verify pass in full production;
  the full-audio/stem readers still validate the exact media they actually read;
- published stage artifacts retain their recorded file hashes;
- resume performs one authoritative artifact recovery/hash validation, not a
  duplicate preflight hash pass;
- source/config/inventory fingerprints remain part of durable identity.

Standalone/frozen paths keep their original stronger repeated verification by
default. The reduced checks apply only to the full-production prevalidated path.

The full downstream supervisor prints:

    shard=<id> downstream_prepare_start
    shard=<id> downstream_prepare_ready rows=... contexts=...
    shard=<id> downstream_preflight_start
    shard=<id> downstream_preflight_ready
    shard=<id> mimo_server_started ...
    shard=<id> mimo_server_ready ...

If GPU utilization is zero for a long period, inspect these markers first. A
stall between downstream_prepare_start and downstream_prepare_ready is inventory
construction / shared-storage work. A stall during downstream_preflight is resume
metadata/artifact recovery. MiMo should not consume GPU memory until
mimo_server_started appears.

### Raw-video server acceptance (2026-09-18)

The earlier performance-v2 raw-5 server run passed with resident upstream pools
and MiMo --mem-fraction-static 0.50. The final stage summary was:

    canonical  ready=5 failed=0
    SAM        records=5 model_call_count=10
    AuK        ready=5 failed=0 model_call_count=5
    resolve    ready=5 failed=0
    DiariZen   jobs=5 ready=5 failed=0
    ASR        jobs=6 ready clips=5 failed=0
    MiMo       t2va_ready=5 ta2va_ready=5, all failed/pending/skipped=0

ASR job_count can exceed ready clip count because ASR jobs are speech segments.
The accepted run exited with no owned worker processes left behind. Subsequent real 2k-shard runs showed that keeping all upstream models resident
caused SAM/AuK OOMs even after lowering MiMo. A later stage-local-upstream run
still produced repeated SAM OOMs while MiMo 0.55 remained resident: SGLang
processes occupied roughly 90-114 GiB per GPU while SAM itself used roughly
25-40 GiB and requested additional 4-12 GiB transient allocations. Production
therefore removed the final overlap as well. MiMo is now lazy per shard and runs
only inside the downstream stage at mem-fraction-static 0.65; it is stopped
before the next shard returns to SAM. This lifecycle preserves the frozen stage
semantics and durable receipts. It is the current production candidate and still
requires server smoke / resumed real-shard validation.

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
  --downstream-run-id mimo-v2.6-flash-rl \
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