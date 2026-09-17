# Target-Side T2VA / TA2VA Production

This orchestration is independent of Visual and writes only its owned production
root. It calls the frozen T2VA v7/backend .7/core .6 and TA2VA .2/profile v2
implementations; the reused audio finalizer remains v6. The standalone downstream
runner does not preprocess Audio. The optional full launcher below adds upstream
stage orchestration without migrating legacy outputs or changing eligibility.

## Full Raw-Video Production

Use scripts/run_h3_t2va_full_production.sh for a fresh, shard-owned Audio
workspace. One node owns one fixed shard through these barriers:

    canonical -> SAM music_first -> AuK -> resolve -> DiariZen -> ASR -> MiMo

MiMo starts once per launcher invocation and stays alive across stages/shards.
The serving arguments are unchanged. Client REQUEST_WORKERS defaults to 1.
Upstream stages use GPU_IDS=0,1,2,3,4,5,6,7 by default. SAM, AuK, DiariZen and
Qwen3-ASR each have eight node-lifetime resident workers (32 upstream workers,
in addition to MiMo). Every worker loads once before its READY handshake,
processes requests across shards and closes only at node shutdown. Stage barriers
remain sequential; residency does not mean concurrent GPU stages. Each child sees
its physical GPU through CUDA_VISIBLE_DEVICES and uses cuda:0; the parent
environment is unchanged. There is no automatic GPU memory manager or restart.

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

The full runner needs the existing server_env.sh dependency settings:
SAM_AUDIO_CODE_ROOT, SAM_AUDIO_MODEL_PATH, SAM_AUDIO_T5_BASE_PATH,
SAM_AUDIO_RUNTIME_PYTHONPATH, AUK_PYTHON, AUK_CODE_ROOT, AUK_CHECKPOINT,
AUK_QWEN_PATH, DIARIZEN_PYTHON, DIARIZEN_CODE_ROOT, DIARIZEN_MODEL_PATH,
QWEN3_ASR_ENV and QWEN3_ASR_MODEL_PATH. No dependency is downloaded. The launcher
does not edit server_env.sh. It adds localhost to NO_PROXY/no_proxy without
removing external proxies. Readiness checks owned PID, /v1/models and /model_info.

```bash
RANK=0 WORLD_SIZE=4 SHARD_SEED=20260917 GPU_IDS=0,1,2,3,4,5,6,7 \
  bash scripts/run_h3_t2va_full_production.sh --dry-run

# Inspect commands first. Explicit opt-in is required for unverified SAM output.
RANK=0 WORLD_SIZE=4 SHARD_SEED=20260917 GPU_IDS=0,1,2,3,4,5,6,7 \
ALLOW_UNVERIFIED=1 bash scripts/run_h3_t2va_full_production.sh

SHARDS=59,12,87 ALLOW_UNVERIFIED=1 \
  bash scripts/run_h3_t2va_full_production.sh
```

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
Only the parent publishes full ordered
inventories. Worker outputs never race on canonical records.jsonl. Successful
receipt/media hashes are checked before reuse; failed jobs retry once on the
next invocation. A worker crash waits for other workers, stops the supervisor,
and leaves unattempted jobs pending. Ctrl-C/TERM terminates owned worker trees.
Never delete ready receipts merely to rerun the stage.

stage_state/<stage>/invocation.json records scheduled_job_count,
reused_ready_count and worker_count for the latest invocation. Zero scheduled
jobs means no inference request is sent; already-resident workers remain idle.
Frozen publication model
counts still describe the cached artifacts, not new calls in this invocation.
Keep interpreters and dependency environments pinned for a resumed run; use a
fresh production root when upgrading runtimes rather than adopting old caches.

Upstream aggregate manifests can expand after failed clips recover. The full
downstream adapter binds ready outputs to validated per-clip dependencies;
unrelated aggregate hash changes cannot trigger repeat inference. Previously
upstream-skipped clips can reopen when their own inputs become available.
Published data remains available to the existing snapshot builder.

### Raw-Video Server Acceptance

The pre-performance raw-5 run passed on the server as reported by the operator.
No real inference was performed for this performance refactor locally. Before
2k production, validate the new lifecycle using fresh roots:

1. Create a new five-row shot JSONL containing the original five target video
   rows, preserving paths and order; do not copy a prior Audio cache.
2. Point SHOT_MANIFEST at that file and PRODUCTION_ROOT at a fresh smoke root.
3. Run SHARDS=0 with the full launcher and ALLOW_UNVERIFIED=1. Verify canonical,
   SAM, AuK, resolved, DiariZen and ASR inventories with their existing loaders.
4. Build a snapshot using tools/build_h3_t2va_snapshot.py. Check the stage and
   downstream ready/failure counters rather than assuming every source is usable.
5. Rerun the identical command. Confirm ready worker receipts were reused and
   ready T2VA/TA2VA artifacts made zero additional model calls.
6. In another fresh smoke root, interrupt during ASR, rerun, and verify only
   unfinished/failed ASR jobs ran. Confirm one MiMo PID per invocation and no
   surviving owned child processes after node shutdown. After each stage, the
   corresponding upstream workers MUST still exist; inspect their PIDs and GPUs.
7. Only after the five-clip checks pass, try a roughly 100-row bounded manifest,
   then one actual 2,000-row shard. Record canonical/SAM/AuK/resolve/DiariZen/
   ASR/MiMo and total wall time.
8. Finally run SHARDS=0,1 on two real 2,000-row shards. Verify identical worker
   PIDs across both shards and exactly one model load per worker. Confirm that
   shard 1 canonical preparation overlaps shard 0 GPU processing. Do not start
   the entire source population before these checks pass.

## Standalone Downstream Prerequisites

The population authority is the original ordered JEA shot JSONL, not an Audio
or Visual inventory. Production shard size is 2,000. Each physical source row belongs permanently to a 2,000-row
shard, including malformed/unavailable rows. Shard 59 is source indexes
118000 through 119999. Shards 0 and 1 cover 0..1999 and 2000..3999.
The last shard can be shorter; shard count is derived from the actual source.

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

## Acceptance Before Large Production

CPU tests cover frozen-vs-production core/prompt equivalence, TA2VA PCM hashes,
resume without repeat calls, concurrent writes, interruption recovery,
incremental snapshots and launcher lifecycle with synthetic executables.
They are not evidence of a real MiMo run.

Before operating the full population, create a separate smoke output root and
a small ordered shot-manifest fixture containing 2-5 rows from the already
validated random20 population, with original video paths unchanged. Use its
matching frozen Audio root/run. Run Python dry-run, then shard 0 with a real
endpoint. Compare:

1. production artifacts/<uid>/t2va/core.json to frozen core semantic payload;
2. T2VA prompt and exact dialogue;
3. TA2VA variant set, six-section prompts and reuse PCM hashes.

Only after confirming no drift, test a bounded small complete shard fixture.
Do not start the full 3.8M source until those server checks pass. This local
development task has not executed real server/model inference.
