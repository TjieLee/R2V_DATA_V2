# Standalone Pre-Qwen Visual Production

Visual Stage 2 is a production wrapper around these frozen Visual V3 stages:

```text
standalone AnnotationState shard
-> exactly 10 sampled frames
-> SAM3 entity segment/tracking
-> temporal coverage/rank
-> deterministic background candidate construction
-> stop before remove
```

It never runs Qwen Annotation, remove, Boogu, a Qwen remove judge, pair,
reference edit, reference integrity, instruction, Subject Attributes, or export.
It directly reuses `frames.py`, `segment.py`, `rank.py`, and `background.py`.

## Qwen-free preflight

Production and canary commands validate the exact base config before loading a
GPU model. They inspect all SAM3 rescue/search modes. If
`multi_instance_rescue_mode=qwen_anchor_select_v1`, startup fails with:

```text
Pre-Qwen Stage2 is not Qwen-free under current frozen SAM3 config.
```

The runner never changes this mode to `off`. The formal production config
documented elsewhere as `e2e1000-s0-samfix-20260814-101818.yaml` uses the Qwen
anchor selector and is intentionally rejected. An explicitly approved frozen
base config with Qwen-free SAM3 policy is required before canary or production.
Successful metadata records `qwen_required=false` and `qwen_calls=0`.

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
  locks/shard-000000000-000009999.lock
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
- Torn JSONL tail: truncate only the incomplete last line.
- Completed shard: validate all rows and durable artifacts before skipping.

Appends flush and fsync. Completion uses atomic rename plus parent-directory
fsync. OS `flock`, not the lock filename, owns a shard; process death releases
ownership automatically.

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
  --base-config /absolute/path/to/qwen-free-frozen-base.yaml \
  --output-root /mnt/workspace/litengjie/data/r2v_v3_stage2/jea_motion_v1/pre-qwen-v1 \
  --gpu 0
```

Dry-run does not load SAM3:

```bash
.venv/bin/python tools/run_v3_pre_qwen_auto.py \
  --input-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations \
  --base-config /absolute/path/to/qwen-free-frozen-base.yaml \
  --output-root /mnt/workspace/litengjie/data/r2v_v3_stage2/jea_motion_v1/pre-qwen-v1 \
  --gpus 0,1,2,3 \
  --dry-run
```

Remove `--dry-run` for single-node four-GPU production. Use
`--gpus 0,1,2,3,4,5,6,7` for eight GPUs. If omitted, the launcher respects
`CUDA_VISIBLE_DEVICES`, then queries local GPUs. Each child sees one GPU as
`cuda:0`, loads one persistent SAM3 backend, and dynamically claims shards.

For multiple nodes, run the same command with the same shared output root.
`RANK`/`WORLD_SIZE` only rotate scan order. Restarting with fewer nodes or GPUs
continues the same logical shards and artifacts.

## Required isolated 8 x 10 canary

Prepare exactly 80 eligible samples—ten per GPU—before formal production:

```bash
.venv/bin/python tools/run_v3_pre_qwen_canary.py \
  --input-root /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations \
  --base-config /absolute/path/to/qwen-free-frozen-base.yaml \
  --output-root /mnt/workspace/litengjie/data/r2v_v3_pre_qwen_canary/preqwen-8x10-test \
  --gpus 0,1,2,3,4,5,6,7 \
  --samples-per-gpu 10 \
  --prepare-only
```

Eligible means `status=ready` with non-empty entities. Preparation writes
`canary_manifest.json` and `selection.jsonl`, verifies selected MP4s, Stage 1
hashes, and Qwen-free policy, and does not load SAM3 or create artifacts.
Repeating the exact command reuses selection; identity drift fails closed.

Remove `--prepare-only` to run. Canary output stays below
`r2v_v3_pre_qwen_canary/<run-id>`, never uses production shard filenames, and
never writes to the formal Stage 2 root. GPU slots get deterministic groups of
ten; canary deliberately does not use production work stealing.

Each GPU process loads SAM3 once and runs its ten clips sequentially.
`workers/gpu-N.jsonl.partial` resumes from the next selected row. `summary.json`
reports per-GPU completion, SAM3/coverage/background counts, retryable failures,
zero Qwen calls, elapsed time, and frame/mask/total bytes for storage sizing.

Real SAM3/GPU execution remains a server validation step and is not performed
by local unit tests.
