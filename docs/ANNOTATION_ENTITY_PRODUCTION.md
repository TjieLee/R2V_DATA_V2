# Entity Annotation Production Runbook

Status: **frozen / in production** as of 2026-08-21.

This document records the standalone entity-annotation path only. Do not use it as a reason to change normal V3 `RunStorage`, Visual thresholds, or the normal writable-root policy.

## 1. Frozen implementation

Repository: `TjieLee/R2V_DATA_V2`

Branch used when production was started: `feature/v3-subject-attributes-v1`

Key commits:

- `8fdce3c513e95baab1e64917ee851c70c5e407c5` — large-scale standalone annotation batch runner
- `8d196c1415b2e72153209361684066c9f8077485` — automatic multi-node / multi-Qwen launcher

Production tools:

```text
tools/run_v3_annotation_batch.py
tools/run_v3_annotation_auto.py
```

The annotation path is now frozen while production is running. Do not casually modify these two files, the annotation schema, or entity-selection semantics.

## 2. Data and output paths

```text
source JSONL:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl

processed clips:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped

source videos:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808

final annotation output:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations
```

No MP4 files are copied by this annotation job. The processed `video_path` is sent to Qwen as the visual input.

The public `entity_annotations` directory is an explicit exception for this standalone production job. It does **not** make `/mnt/workspace/public/dataset` a general writable root for normal V3 runs.

## 3. Sharding and restart behavior

Fixed logical shard size:

```text
10000 source rows / shard
```

Examples:

```text
shard 0 = source_index 0        ... 9,999
shard 1 = source_index 10,000   ... 19,999
shard N = N*10000               ... (N+1)*10000-1
```

The final shard may be partial.

Each source row produces one output row, including failures. There are no silent omissions.

Output layout:

```text
entity_annotations/
  parts/
    shard-000000000-000009999.jsonl
    shard-000010000-000019999.jsonl
    ...
  failures/
  locks/
  logs/
```

Restart behavior is local to the annotation shard and intentionally independent of the repository Git HEAD:

- completed `.jsonl` shard: validate and skip
- `.jsonl.partial`: validate completed contiguous rows and continue from the next row
- each appended row is flushed/fsynced
- completion uses atomic rename from `.partial` to `.jsonl`
- shard lock prevents accidental concurrent ownership of the same shard on a shared filesystem

Changing Git commits later must not block continuation of this standalone annotation production job.

## 4. Multi-node / 8-GPU-per-node execution

The production wrapper reuses the scheduler-provided environment variables:

```bash
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
```

If the cluster scheduler already injects different `RANK` values and the same `WORLD_SIZE` on all nodes, operators do not manually enter node ranks.

Every node uses all eight local GPUs as independent TP1 Qwen servers:

```text
GPU 0 -> 8001
GPU 1 -> 8002
GPU 2 -> 8003
GPU 3 -> 8004
GPU 4 -> 8005
GPU 5 -> 8006
GPU 6 -> 8007
GPU 7 -> 8008
```

Model:

```text
/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
```

This replaces the colleague pipeline's old `Qwen3.8-27B-FP8` entity-word extraction path. The old `step1_get_entity_words.py` path is not part of this production flow.

Typical vLLM launch block used by the external `stage1_run.sh` wrapper:

```bash
MODEL=/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct

for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup vllm serve "$MODEL" \
    --port $((8001+i)) \
    --tensor-parallel-size 1 \
    --max-model-len 49152 \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --allowed-local-media-path /mnt/workspace/public/dataset \
    > "$LOG_DIR/vllm_$((8001+i)).log" 2>&1 &
done

BASE_URLS=""
for port in $(seq 8001 8008); do
  until curl -sf "http://127.0.0.1:${port}/v1/models" > /dev/null; do sleep 10; done
  BASE_URLS="${BASE_URLS}${BASE_URLS:+,}http://127.0.0.1:${port}/v1"
done
```

Then start the R2V automatic launcher:

```bash
/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python \
  /mnt/workspace/litengjie/data/R2V_DATA_V2/tools/run_v3_annotation_auto.py \
  --base-urls "$BASE_URLS" \
  --rank "$RANK" \
  --world-size "$WORLD_SIZE"
```

The intended operator experience on a prepared node is therefore simply:

```bash
bash stage1_run.sh
```

`stage1_run.sh` belongs to the surrounding data-pipeline environment; the R2V repository owns the two Python tools listed above.

## 5. Automatic work allocation

`run_v3_annotation_auto.py` performs the allocation automatically:

1. Count complete, non-empty rows currently present in `shots_f03_motion.jsonl`.
2. Compute `ceil(rows / 10000)` logical shards.
3. Split the global shard range across `WORLD_SIZE` nodes using `RANK`.
4. Split the current node's shard range across the supplied local Qwen URLs.
5. Generate a rank-local YAML under `/mnt/workspace/litengjie/data/entity_annotation_configs`.
6. Start one sequential annotation worker per Qwen URL.

A single Qwen endpoint processes one clip request at a time. Parallelism comes from independent endpoints/processes.

The source JSONL can grow. The launcher snapshots the current complete-row count when it starts. If new source rows are appended later, rerun the same production wrapper after the current allocation is complete; completed shards will be skipped and newly created shards will be picked up.

## 6. Useful commands

Inspect a completed shard:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2

.venv/bin/python tools/run_v3_annotation_batch.py \
  --inspect \
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations/parts/shard-000000000-000009999.jsonl
```

The inspector reports:

```text
rows
ready
failed
total_entities
subject
object
group
```

Check annotation workers:

```bash
ps -ef | grep '[r]un_v3_annotation_batch.py'
```

Rank-local launcher logs:

```text
/mnt/workspace/litengjie/data/entity_annotation_logs/rank-<RANK>/
```

## 7. Freeze boundary

While production is active:

- do not change annotation output schema or entity-selection semantics
- do not change shard size from 10000 for this production dataset
- do not reintroduce normal V3 `RunStorage` git-commit identity into the standalone annotation runner
- do not loosen normal V3 public-dataset write protection
- do not mix Audio/H3 changes into this branch for annotation work
- do not replace the current Qwen3-VL model path with the colleague pipeline's old Qwen3.8 model

If a production defect requires a change, treat it as an explicit migration/versioning decision rather than an opportunistic refactor.