# Target-Side T2VA / TA2VA Production

This orchestration is independent of Visual and writes only its owned production
root. It calls the frozen T2VA v7/backend .7/core .6 and TA2VA .2/profile v2
implementations; the reused audio finalizer remains v6. It does not preprocess
Audio, migrate legacy outputs, or change semantic eligibility.

## Prerequisites and Population

The population authority is the original ordered JEA shot JSONL, not an Audio
or Visual inventory. Each physical source row belongs permanently to a 10,000-row
shard, including malformed/unavailable rows. Shard 59 is source indexes
590000 through 599999. The last shard can be shorter.

The selected clips need matching, finalized canonical Audio and a named
resolved AuK/SAM -> DiariZen -> Qwen3-ASR cache. Missing upstream data retains the
existing skipped semantics; this runner does not launch those models. Supply an
Audio root/run covering the intended production population. Do not point a
full-population run at a random20-only cache and expect Audio to be generated.
Skipped clips are not automatically retried; use a new production root when
changing the source/config/upstream population.

This checkout has no tools/export_h3_training_manifests.py or independent
target-side T2V producer. The existing Visual t2v_caption field is not a separate
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
