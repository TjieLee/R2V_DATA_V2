# Standalone Pre-Qwen Visual Production

Visual Stage 2 is a production wrapper around these frozen Visual V3 stages:

```text
standalone AnnotationState shard
-> exactly 10 sampled frames
-> SAM3 entity segment/tracking
   -> frozen QwenSam3AnchorSelector on multi-instance ambiguity
-> temporal coverage/rank
-> deterministic background candidate construction
-> stop before remove
```

It never runs Qwen Annotation, remove, Boogu, a Qwen remove judge, pair,
reference edit, reference integrity, instruction, Subject Attributes, or export.
It directly reuses `frames.py`, `segment.py`, `rank.py`, and `background.py`.
Here “Pre-Qwen” means stopping before the next standalone/downstream Qwen
business stage (the Remove Judge); it does not mean zero Qwen calls. The
targeted `QwenSam3AnchorSelector` remains part of the frozen SAM3 algorithm.

## Frozen SAM3 and candidate-judge preflight

Production and canary commands validate the exact base config before loading a
GPU model. The production config keeps:

```yaml
sam3:
  multi_instance_rescue_mode: qwen_anchor_select_v1
qwen:
  candidate_judge:
    base_url: http://6.167.57.88:8000/v1
```

The runner neither disables nor reimplements this mode. It requires a valid
`qwen.candidate_judge` service and performs a short `GET /models` availability
check before loading SAM3. An unavailable gateway fails startup; there is no
fallback to `off`. Metadata records the SAM3 rescue mode, logical candidate
judge endpoint/model, and zero calls only for Annotation and Background Remove
Qwen stages, which Stage2 does not run.

The Qwen service node hosts eight TP1 replicas on ports 8001 through 8008 behind
one OpenAI-compatible gateway at port 8000. The gateway should use least
connections and health-check its replicas. Every SAM worker on every node uses
the same logical `http://6.167.57.88:8000/v1` endpoint. GPU/rank topology never
selects a replica or changes the durable config identity.

## Input and logical shard identity

Only completed Stage 1 files below `entity_annotations/parts/*.jsonl` are
inputs. `.partial`, `failures`, `locks`, and `logs` are ignored. Every Stage 1
row produces exactly one Stage 2 row. Annotation failures and ready annotations
without entities are committed as skips without frame decoding or SAM3.

The Stage 1 filename is immutable production identity:

```text
Stage 1: shard-000000000-000009999.jsonl
Stage 2: shard-000000000-000009999.jsonl
```

Machine count, GPU count, `RANK`, `WORLD_SIZE`, hostname, and worker name never
enter filenames, artifact identity, metadata identity, or resume identity.
Final partial-sized Stage 1 shards keep their nominal filename boundaries.

The canonical logical shard and the execution scheduling unit are deliberately
different:

```text
canonical logical shard: 10,000 Stage 1 source rows
execution chunk:            100 source rows by default
```

Chunk ranges are derived only from the logical shard's contiguous
`source_index` rows. GPU count, node count, eligible-row count, and runtime
speed do not change chunk identity. A short final logical shard may have a
short final chunk.

```text
Stage 1 logical shard
        ↓
fixed source-row execution chunks
        ↓
dynamic nonblocking multi-node/GPU claim
        ↓
existing frozen per-clip Visual pipeline
        ↓
durable chunk fragments
        ↓
deterministic validated compaction
        ↓
same canonical Stage 2 logical shard
```

Chunks are an internal execution detail and are never part of the Stage 3 data
contract.

## Production output and schema

Recommended writable root:

```text
/mnt/workspace/litengjie/data/r2v_v3_stage2/jea_motion_v1/pre-qwen-v1
```

Layout:

```text
pre-qwen-v1/
  parts/
    shard-000000000-000009999.jsonl
    shard-000000000-000009999.jsonl.partial
    shard-000000000-000009999.meta.json
  _internal/
    execution.json
    shard-000000000-000009999/chunks/
      chunk-000000000-000000099.jsonl
      chunk-000000100-000000199.jsonl.partial
  locks/
    shard-000000000-000009999/
      chunk-000000000-000000099.lock
    compact-shard-000000000-000009999.lock
  failures/shard-000000000-000009999.jsonl
  artifacts/
    shard-000000000-000009999/<clip_uid>/
      state.json
      run/
        run.json
        clips/<clip_uid>/
          clip.json
          frames/00.jpg ... 09.jpg
          frames/frames.json
          masks.rle.json
          background/source_mask_<sha256>.png
  logs/
```

Each clip is a RunStorage-compatible workspace. `clip.json` durably contains
the imported `AnnotationState`, `CoverageState`, and `BackgroundReferenceState`.
Stage 3 can hydrate these canonical artifacts without rerunning Annotation,
frame decode, SAM3, coverage, or background selection.

The row schema is `r2v.v3.pre_qwen_stage2.1`. Terminal statuses are:

```text
skipped_annotation_failed
skipped_no_entities
coverage_rejected
ready_no_background
ready_background_rejected
ready_background_pending_remove
failed_input
failed_frames
```

Coverage rejection and background `none`/`rejected` are business outcomes.
The `_internal` tree is retained for crash audit and production validation;
downstream readers use only the canonical `parts/` and `artifacts/` interface.

## Resume, transactions, and retryable failures

`state.json` records:

```text
annotation_ready
frames_ready
masks_ready
coverage_ready
background_ready
row_committed
```

Resume revalidates the Stage 1 shard SHA-256, source prefix, clip UID, row hash,
base-config byte hash and fingerprint, Visual freeze, frames, masks, recomputed
coverage, background state, and source mask.

- `frames_ready`: validate all ten images, then run SAM3.
- `masks_ready`: validate frames/masks, then compute coverage.
- `coverage_ready`: recompute and compare coverage, then build background.
- Complete background before row append: validate and reuse it.
- Torn chunk JSONL tail: truncate only the incomplete last line.
- Completed shard: validate all rows and durable artifacts before skipping.

Each chunk append flushes and fsyncs. Chunk completion atomically renames its
`.partial` file. Once every chunk is complete, one nonblocking shard-compaction
lock validates every row against the Stage 1 shard, writes and fsyncs the
canonical shard `.partial`, and atomically publishes the unchanged canonical
filename. Duplicate, missing, reordered, wrong-index, or wrong-hash rows fail
closed. OS `flock`, not a lock filename, owns a chunk; process death releases
ownership automatically and another worker can resume its durable prefix.

`_internal/execution.json` freezes the execution chunk schema and chunk size.
Restarting an existing chunk run with another `--chunk-rows` value fails
closed. An unfinished legacy whole-shard `.partial` is never silently
reinterpreted as chunk output; use the previous runner to finish it or choose a
new output root. Completed legacy canonical shards remain readable and
validatable.

CUDA OOM/context errors, SAM3 exceptions, worker death, and temporary IO
failures do not append terminal rows. A retryable diagnostic is written and the
shard remains partial at that source row. Confirmed invalid input or
deterministic frame decode failure may publish a fail-closed input/frame result.

## Production commands

Inspect one shard:

```bash
.venv/bin/python tools/run_v3_pre_qwen_batch.py \
  --inspect /mnt/workspace/litengjie/data/r2v_v3_stage2/jea_motion_v1/pre-qwen-v1/parts/shard-000000000-000009999.jsonl
```

Run one shard on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools/run_v3_pre_qwen_batch.py \
  --input-shard /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations/parts/shard-000000000-000009999.jsonl \
  --base-config /absolute/path/to/frozen-production-base.yaml \
  --output-root /mnt/workspace/litengjie/data/r2v_v3_stage2/jea_motion_v1/pre-qwen-v1 \
  --gpu 0 \
  --chunk-rows 100
```

Dry-run does not load SAM3:

```bash
.venv/bin/python tools/run_v3_pre_qwen_auto.py \
  --input-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations \
  --base-config /absolute/path/to/frozen-production-base.yaml \
  --output-root /mnt/workspace/litengjie/data/r2v_v3_stage2/jea_motion_v1/pre-qwen-v1 \
  --gpus 0,1,2,3 \
  --dry-run
```

Remove `--dry-run` for single-node four-GPU production. Use
`--gpus 0,1,2,3,4,5,6,7` for eight GPUs. If omitted, the launcher respects
`CUDA_VISIBLE_DEVICES`, then queries local GPUs. Each child sees one GPU as
`cuda:0`, loads one persistent SAM3 backend, and dynamically claims 100-row
chunks. The runtime SAM3 backend still requests `device=cuda`; the process-level
`CUDA_VISIBLE_DEVICES` mapping owns the physical GPU.

For multiple nodes, run the same command with the same shared output root.
`RANK`/`WORLD_SIZE` only rotate scan order. Every worker scans all unfinished
chunks with wraparound, skips busy locks without waiting, and immediately
claims another chunk after completion. Restarting with fewer or more nodes or
GPUs continues the same chunks, checkpoints, logical shards, and artifacts.
When a full scan finds only temporarily busy work, the worker sleeps for one
second and rescans while retaining its persistent SAM3 backend. The default
`--idle-exit-seconds 60` grace permits reclaim after a worker/node crash while
allowing surplus workers to exit after 60 continuous seconds without a claim;
any successful claim resets that timer. No worker waits on a specific lock.
The normal launcher performs only cheap shard enumeration plus config/Qwen
health preflight before spawning workers; `--dry-run` retains the full
inventory scan.

Stage2 can optionally overlap the unchanged CPU frame stage with persistent
SAM3 workers by adding `--frame-prefetch-workers 32` to the auto-launcher.
The option defaults to `0`, so existing execution is unchanged unless it is
explicitly enabled. One node-level `ProcessPoolExecutor` scans ready rows with
non-empty entities and calls the frozen `_build_clip_frames` path for each
clip. It writes the same workspace, ten JPEG frame artifact, and existing
`frames_ready` checkpoint; it does not write Stage2 rows or run SAM3, coverage,
background, or Qwen. A filesystem `flock` per clip serializes prefetch and SAM
frame preparation across processes and nodes. SAM workers re-check under that
lock, reuse validated prefetched frames when available, and otherwise retain
the synchronous frame-build fallback. After GPU workers finish, the launcher
stops and joins the CPU pool so no prefetch child remains orphaned. Individual
prefetch failures are diagnostic only and are recorded in
`logs/frame-prefetch.jsonl`; canonical Stage2 processing remains authoritative.
The launcher summary reports `frame_prefetch_workers`, submitted, completed,
skipped-existing, failed, and wall-seconds counters.

For an opt-in SAM3 runtime timing diagnostic, add
`--sam3-request-timing`. The flag defaults off and only wraps the predictor
request interface in each Stage2 worker; it does not add, remove, reuse, or
modify any SAM3 request or session. `handle_request` timings are grouped by
request type, while streaming propagation is timed through complete iterator
consumption. Each worker reports the frozen backend performance, anchor-search,
and recall-rescue counters alongside the request timings. The auto-launcher
aggregates start-session, add-prompt, propagation, close-session, total-track,
request-fraction, and non-negative unattributed wall time, while retaining a
per-GPU map. For the isolated request-timing benchmark, keep frame prefetch
disabled so the invocation includes:

```bash
--frame-prefetch-workers 0 --sam3-request-timing
```

This is diagnostic instrumentation only; it does not implement session reuse,
feature caching, warmup requests, or another SAM3 optimization.

## Required isolated 8 x 10 canary

Prepare exactly 80 eligible samples—ten per GPU—before formal production:

```bash
.venv/bin/python tools/run_v3_pre_qwen_canary.py \
  --input-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations \
  --base-config /absolute/path/to/frozen-production-base.yaml \
  --output-root /mnt/workspace/litengjie/data/r2v_v3_pre_qwen_canary/preqwen-8x10-qwen-anchor-20260828 \
  --gpus 0,1,2,3,4,5,6,7 \
  --canary-shards 16 \
  --samples-per-shard 5 \
  --prepare-only
```

Selection uses the first 16 completed canonical Stage 1 shards and the first
five eligible rows in `source_index` order from each shard. Eligible means
`status=ready` with non-empty entities, a unique `clip_uid`, and a valid
processed MP4. A short selected shard, duplicate cross-shard `clip_uid`, or a
total not divisible by the GPU count fails closed; later shards never backfill
the quota. Preparation writes `canary_manifest.json` and `selection.jsonl`,
verifies selected MP4s, Stage 1 hashes, and frozen SAM3/Qwen-anchor policy, and
does not load SAM3 or create artifacts. Repeating the exact command reuses the
persisted selection; identity drift fails closed.

Remove `--prepare-only` to run. Canary output stays below
`r2v_v3_pre_qwen_canary/<run-id>`, never uses production shard filenames, and
never writes to the formal Stage 2 root. GPU slots get deterministic groups of
ten; canary deliberately does not use production work stealing.

Each GPU process loads SAM3 once and runs its ten clips sequentially.
`workers/gpu-N.jsonl.partial` resumes from the next selected row. A retryable
worker failure keeps that partial state and makes the top-level command exit
nonzero. `summary.json` separates row completion (`status`) from
`functional_status`: functional pass requires all 80 selections complete, no
outstanding retryable work, and zero `failed_input` or `failed_frames` rows.
Coverage rejection and missing/rejected background are valid business outcomes.
The summary reports SAM3/coverage/background counts, process-delta anchor/rescue
counters, logical candidate-judge identity, elapsed time, and frame/mask/total
bytes for storage sizing. Annotation and Background Remove Qwen call counts stay
zero; targeted SAM3 anchor Qwen calls are represented by frozen profiling and
multi-instance rescue counters.

The earlier `preqwen-8x10-20260828` run used
`multi_instance_rescue_mode=off`. Preserve it as runner/checkpoint evidence, but
do not treat or resume its masks as frozen production-semantics results. Use the
new output root above; the deterministic 16-by-5 selection produces the same 80
samples for A/B comparison while the changed config identity prevents reuse.

Real SAM3/GPU execution remains a server validation step and is not performed
by local unit tests.
