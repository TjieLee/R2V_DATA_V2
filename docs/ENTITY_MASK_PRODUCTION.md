# Entity Mask Stage2 Production Runbook

Last updated: 2026-08-31

This runbook covers the standalone Visual Stage2 production job only. It
consumes frozen Stage1 entity annotations and stops before foreground removal.
The formal production scheduler is static whole-shard ownership; the older
dynamic cross-node chunk-claim scheduler is retained only as a generic Stage2
runner and is not the Entity Mask production path.

## Production identity

```text
repository:
TjieLee/R2V_DATA_V2

production branch:
feature/v3-sam3-production-v1

validated launcher HEAD:
8645db869892ea98a6599467882efc48b4eb7414

frozen Visual algorithm:
d056c32b76db4b3d7c0358b38e996e7a91a288d1
```

The recent production commits are orchestration/runtime hardening only. They do
not modify the frozen Visual algorithm files or the public Visual/Audio
production schema.

## Production contract

```text
input:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_annotations

output:
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/entity_mask

private validated config:
/mnt/workspace/litengjie/data/entity_mask_configs/production.yaml

private worker logs:
/mnt/workspace/litengjie/data/entity_mask_logs

SAM3 source:
/mnt/workspace/litengjie/data/vendor/sam3

SAM3 checkpoint:
/mnt/workspace/public/pretrained/facebook/sam3/sam3.pt
```

The output root is an exact standalone-production exception. It does not make
the public dataset tree generally writable, and normal V3 `V3Config` and
`RunStorage` public-root protection remains in force.

Prepare the private config once from the server-validated Stage2 config:

```bash
mkdir -p /mnt/workspace/litengjie/data/entity_mask_configs
cp /mnt/workspace/litengjie/data/stage2_no_diagnostics.yaml \
  /mnt/workspace/litengjie/data/entity_mask_configs/production.yaml
```

The launcher fails closed if this file is missing or if its frozen production
semantics do not match diagnostics-off, progressive SAM3 rescue, and the
central Qwen candidate judge configuration.

## Required multi-node environment

Every physical node must see the same repository, production config, Stage1
annotations, Entity Mask output root, SAM3 source tree, and SAM3 checkpoint.
Using the same `.venv/bin/python` path is not sufficient by itself: the SAM3
source root must also be importable on every node.

The production shell must include:

```bash
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29506}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export TORCH_HOME=/mnt/workspace/public/.cache/

source /mnt/workspace/liutao/miniconda3/etc/profile.d/conda.sh
conda activate dreamidv

export CUDA_HOME=/usr/local/cuda
export CPATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/include:${CPATH:-}
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/mnt/workspace/litengjie/data/vendor/sam3${PYTHONPATH:+:$PYTHONPATH}
```

Before a multi-node run, verify on every node:

```bash
/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python - <<'PY'
from sam3.model_builder import build_sam3_video_predictor
print("SAM3_IMPORT_OK")
PY
```

All nodes must print `SAM3_IMPORT_OK`. A missing `PYTHONPATH` or a node that
cannot see `/mnt/workspace/litengjie/data/vendor/sam3` fails later as
`ModuleNotFoundError: No module named 'sam3'`.

## External production entry

The operator entry remains:

```bash
bash stage_entity_mask_run_v2.sh
```

The shell invokes the zero-argument launcher:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2

/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python \
  /mnt/workspace/litengjie/data/R2V_DATA_V2/tools/run_v3_entity_mask_auto.py
```

No production arguments are required. `RANK` and `WORLD_SIZE` are supplied by
the cluster environment and default to `0` and `1` for a single node. The
launcher auto-detects visible local GPUs and starts one persistent SAM3 worker
per GPU.

The targeted ambiguity judge uses the centralized gateway:

```text
http://6.167.57.88:8000/v1
/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
```

Startup performs the candidate-judge health check. There is no fallback that
disables the frozen Qwen anchor selector.

## Static whole-shard parallelism

The canonical Stage1 logical shard is 10,000 source rows. The formal Entity Mask
scheduler assigns each complete logical shard to exactly one global GPU worker:

```text
global_worker_count = WORLD_SIZE * visible_local_gpu_count
global_worker_id    = RANK * visible_local_gpu_count + local_gpu_slot
```

Sorted Stage1 shards are divided with deterministic contiguous balanced
partitioning. If `N` shards are divided over `W` workers, each worker receives
`floor(N/W)` shards and the first `N mod W` workers receive one additional
shard. There is no overlap, gap, dynamic stealing, or cross-node reassignment.

Example from the validated single-node dry-run:

```text
Stage1 shards: 384
WORLD_SIZE:    1
local GPUs:    8
global workers: 8
assignment:   48 complete 10k shards per GPU
```

The exact number of nodes is never hard-coded. A different cluster allocation
changes `WORLD_SIZE`, and the launcher computes the global worker topology at
runtime.

Each worker performs:

```text
load one SAM3 backend
    -> shard A
    -> shard B
    -> shard C
    -> ...
close backend once
```

SAM3 is not reloaded per shard.

## Durable execution inside one 10k shard

Static ownership changes scheduling only. The durable data model remains:

```text
10,000-row logical shard
    -> 100-row execution chunks
    -> per-clip checkpoint/resume
    -> validated chunk publication
    -> canonical 10,000-row compaction
```

The 100-row execution chunks are internal recovery units, not cross-GPU work
units. They preserve restartability without allowing two workers to own one
logical shard.

`state.json` stages remain:

```text
annotation_ready
frames_ready
masks_ready
coverage_ready
background_ready
row_committed
```

Completed canonical shards are validated and skipped. Valid partial chunks and
per-clip checkpoints resume. Corrupt or provenance-mismatched state fails
closed; production code does not silently delete or repair it.

## Locking model

The old dynamic scheduler let all workers scan all chunks and used shared
filesystem `flock` for chunk ownership. A real multi-node production attempt on
2026-08-29 produced workspace-level `FileNotFoundError` failures involving
`run.json` and `masks.rle.json` while multiple nodes shared the same output
root. The exact distributed-filesystem failure mode was not proven, but the
production design was changed so correctness no longer depends on cross-node
work stealing or per-shard/per-clip lock ownership.

In static Entity Mask production:

```text
metadata lock:    disabled for the static owner
frame lock:       disabled for the static owner
chunk claim lock: disabled
compaction lock:  disabled
```

The only remaining `flock` in the normal static production lifecycle is rank 0
startup initialization of the shared execution/session identity marker through
`execution.lock`. It is not used for 100-row, clip, shard, or compaction work.

The generic dynamic Stage2 runner keeps its original lock semantics. Do not
remove those locks globally.

## Startup identity and topology freeze

Rank 0 initializes and validates:

```text
_internal/execution.json
_internal/sam3-session-reuse.json
_internal/entity-mask-static-assignment.json
```

Other ranks wait up to 120 seconds for the complete startup identity. Temporary
`FileNotFoundError` caused by shared-filesystem metadata visibility is retried;
JSON/schema/topology/config mismatches fail immediately.

The static topology marker freezes at least:

```text
WORLD_SIZE
local_gpu_count
global_worker_count
ordered Stage1 shard-list hash
chunk_rows = 100
sam3_session_reuse_mode = clip_reset_v1
```

Therefore a root first initialized with `WORLD_SIZE=1` must not later be reused
for a multi-node topology. Single-node smoke output should be archived as a
whole root before formal multi-node production. Do not delete only the topology
marker to bypass this protection.

## Worker lifecycle and failure semantics

A worker failure does not fail-fast the other fixed-ownership workers. Healthy
workers finish their own shard ranges, then the launcher emits the complete
worker exit-code list and exits nonzero if any worker failed.

Worker return codes are normalized as follows:

```text
positive child code -> same nonzero parent code
-SIGKILL (-9)       -> 137
-SIGTERM (-15)      -> 143
```

The launcher temporarily handles `SIGTERM` and `SIGINT`. On launcher
interruption or launcher-side exception it terminates only the children it
started, uses one shared 10-second grace deadline, escalates remaining children
to `SIGKILL`, reaps them, and restores the original signal handlers. This
prevents orphan static workers from surviving and later colliding with a
restarted launcher that owns the same shard ranges.

To stop a node manually, terminate its launcher first:

```bash
pkill -TERM -f 'run_v3_entity_mask_auto.py'
```

Normally the launcher cleans its workers. As a fallback after an externally
forced launcher death such as `SIGKILL`:

```bash
pkill -TERM -f 'run_v3_entity_mask_worker.py'
```

Run these commands only for the intended node/job. The Python launcher itself
never uses `pkill`, `killall`, or command-string process discovery.

## Fixed production settings

```text
execution chunk rows:       100
frame prefetch workers:     0
SAM3 request timing:        disabled
SAM3 session reuse:         clip_reset_v1
debug diagnostics:          disabled
binary-mask RLE:            vectorized NumPy implementation
logical output shard rows:  10000
```

The accepted production optimizations are deliberately outside the frozen
Visual algorithm:

1. Disable debug overlays/diagnostic image generation.
2. Reuse one physical SAM3 session per clip with `clip_reset_v1` while
   preserving frozen logical session boundaries.
3. Use the vectorized NumPy binary-mask RLE codec with byte-identical serialized
   mask semantics.
4. Use persistent one-backend-per-GPU workers.
5. Use static complete-shard ownership for formal multi-node Entity Mask
   production.

The measured optimization sequence on the same 80-row Stage2 workload was:

```text
original diagnostics-enabled baseline: 5m11.471s
session reuse + diagnostics off:        3m50.727s
vectorized RLE production path:         1m43.285s
wall-time reduction:                    ~66.8%
throughput improvement:                 ~3.02x
```

The real mask A/B compared 54 `masks.rle.json` artifacts with zero differences.
A later canonical verification also confirmed byte-identical canonical JSONL
rows between the scalar/vectorized execution paths; metadata was semantically
identical apart from expected run timestamps.

See `V3_STAGE2_E2E_PERFORMANCE_OPTIMIZATIONS.md` for the benchmark details.

## Lightweight dry-run

The formal Entity Mask `--dry-run` validates config/topology and computes only
the static assignment plan. It does not load SAM3, check Qwen health, initialize
startup identity, scan all annotation rows, or inventory the artifact tree.

```bash
/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python \
  /mnt/workspace/litengjie/data/R2V_DATA_V2/tools/run_v3_entity_mask_auto.py \
  --dry-run
```

The startup plan prints:

```text
rank
world_size
local_gpu_count
global_worker_count
stage1_completed_shards
per-GPU global_worker_id
per-GPU shard start/end positions
per-GPU shard count
```

A normal production run emits `event="startup_plan"` before spawning workers
and `event="completed"` after all workers exit.

## Frozen files

Do not modify the frozen Visual algorithm to change production orchestration:

```text
r2v_data_v2/v3/frames.py
r2v_data_v2/v3/segment.py
r2v_data_v2/v3/sam3_backend.py
r2v_data_v2/v3/sam3_anchor_selector.py
r2v_data_v2/v3/rank.py
r2v_data_v2/v3/background.py
r2v_data_v2/v3/mask_codec.py
```

The vectorized `mask_codec.py` is already accepted production code; do not
restore the old per-pixel Python implementation.

## Audio/H3 branch impact

The static Entity Mask work is Visual Stage2 orchestration. The recent commits
from the first Entity Mask production launcher through lifecycle hardening touch
`pre_qwen_production.py`, Entity Mask launchers, tests, and Visual production
docs. They do not modify `r2v_data_v2/h3/*`, Audio/H3 tools, or the stable
`r2v.v3.production_sample.1` integration contract.

Audio/H3 remains on its own branches and is not directly changed unless these
Visual commits are explicitly merged/cherry-picked. Read
`V3_VISUAL_AUDIO_INTEGRATION.md` before future cross-branch integration.
