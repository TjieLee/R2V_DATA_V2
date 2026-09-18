# Text-only H3-PDD 2/4/8-GPU executor

`pair` is the historical file/API name. Execution groups support
`--group-size {2,4,8}` (default 2); all sizes use the same worker and durable state.
The visible GPU count must equal group-size before any worker is launched.
Group size is execution-only, excluded from input/case identity; the existing
`text_two_person_pdd_fsdp2_pair_v1` identity contract is unchanged. Prepared and
DONE artifacts from a two-GPU run remain reusable with four/eight GPUs.

## Optional hybrid context parallelism

`--ulysses-degree {1,2,4,8}` defaults to 1 (unchanged pure FSDP path).
It must divide group-size and cannot exceed it. Neither degree changes durable
identity. Qwen workers/partitions still depend only on group-size. The node
launcher forwards `--ulysses-degree` unchanged; no new scheduler is introduced.

| Group | Ulysses | FSDP | Mesh |
|---|---|---|---|
| 4 | 1 | 4 | `(4,)`, `fsdp` (baseline unchanged) |
| 4 | 2 | 2 | `(1,2,2)`, `ring,ulysses,fsdp` |
| 4 | 4 | 1 | `(1,4,1)` |
| 8 | 2 / 4 / 8 | 4 / 2 / 1 | `(1,CP,FSDP)` |

Pinned Diffusers `d035dcd7cc7c88e0a154609b62887d50bba9fdc2` was re-inspected:
H3's `_cp_plan` splits packed hidden states, RoPE, adaln/timestep indices and
gathers the PDD-replaced proj_out/audio_proj_out outputs. The official PDD head
retains three-dimensional output shape. The worker passes **no custom cp_plan**
and does not rewrite attention. Ordering is **apply/validate PDD → enable built-in
CP on transformer_ref → FSDP2 on mesh['fsdp'] only**. CP gets the complete custom
mesh with ring_degree=1 and ulysses_anything=True. Text encoder has no CP and
uses only the same FSDP submesh. At FSDP degree 1, transformer/conditioner stay
replicated without FSDP or offload; loading OOM is not worked around.

No attention-backend option/download is added in this patch. The first test
uses the current default attention backend. CPU structural tests cover call
ordering/submesh/step-arm preservation, not real collective/hook compatibility.

### Safe case05 benchmark preparation (never overwrite DONE)

User-reported FSDP4 baseline: **1573.101990563795 s (26.2184 min)**,
311 frames, 1568x672, seed 42, 8 NFE; all four ranks allocated 95.92 GiB /
reserved 115.36 GiB; one successful job and one model/PDD/distributed init.
This is server-provided evidence, not a local measurement.

Do not rerun over its successful root. Using the existing CASE05_ARGS shell
array (including its exact resource paths), set a **new nonexistent** root:

```bash
export CP_ROOT=/mnt/workspace/litengjie/data/person_replacement/h3_case05_cp2_fsdp2_v1
"$PAIR_PYTHON" - "${CASE05_ARGS[@]}" <<'PY'
import copy
import os
import sys
from pathlib import Path
from tools.person_replacement.run_h3_pdd_pair_executor import arguments
from r2v_data_v2.person_replacement.h3_pair_executor import make_config, validate_identity
from r2v_data_v2.person_replacement.h3_pair_state import atomic_bytes, read_json

old_args = arguments(sys.argv[1:])
old_root, old = make_config(old_args)
validate_identity(old_root, old, read_only=True)
new_args = copy.copy(old_args)
new_args.output_root = Path(os.environ['CP_ROOT'])
new_args.group_size, new_args.ulysses_degree = 4, 2
new_root, new = make_config(new_args)
if new_args.output_root.exists() or old['identity'] != new['identity'] or len(old['cases']) != 1:
    raise RuntimeError('Need a new root and the identical single-case input/config identity')
source_case, target_case = old['cases'][0], new['cases'][0]
prep = Path(source_case['directory'])/'preparation'
payload = read_json(prep/'prepared.json')
expected = dict(case_id=target_case['case_id'], row_sha256=target_case['row_sha256'],
                identity=new['identity'], frames=311, width=1568, height=672, seed=42)
if any(payload.get(k) != v for k, v in expected.items()):
    raise RuntimeError('Preparation identity/geometry mismatch')
original = Path(source_case['directory'])/'original.mp4'
if original.resolve(strict=True) != Path(payload['source']).resolve(strict=True):
    raise RuntimeError('Source mismatch')
names = ['source_subject_1','source_subject_2','shot_description',
         'replacement_subject_1','replacement_subject_2','h3_prompt']
files = {name+'.txt': (prep/(name+'.txt')).read_bytes() for name in names}
if files['h3_prompt.txt'] != (payload['prompt']+'\n').encode('utf-8'):
    raise RuntimeError('Prompt bytes mismatch')
files['prepared.json'] = (prep/'prepared.json').read_bytes()
validate_identity(new_root, new)
target = Path(target_case['directory'])
for name, data in files.items():  # prepared marker LAST, exact bytes, no generation/history
    atomic_bytes(target/'preparation'/name, data)
    if (target/'preparation'/name).read_bytes() != data:
        raise RuntimeError('Copy verification failed')
(target/'original.mp4').symlink_to(payload['source'])
print('Exact preparation copied; no generation/attempts/failures/sessions copied')
PY

CUDA_VISIBLE_DEVICES=0,1,2,3 \
"$PAIR_PYTHON" "$PAIR_CLI" "${CASE05_ARGS[@]}" \
  --output-root "$CP_ROOT" --resume --group-size 4 --ulysses-degree 2
```

The last output-root option selects the new root; the original DONE output is
untouched. Verify no Qwen run and byte-identical prompt/preparation. First accept
CP2×FSDP2 before any CP4×FSDP1 experiment (use another new root for CP4).

Manifest/stats include model_load_wall_seconds, pdd_apply_wall_seconds,
parallel_setup_wall_seconds, reference_prepare_wall_seconds,
pipeline_infer_wall_seconds, encode_wall_seconds, total_case_wall_seconds and
parallel_setup_count. Inference timing synchronizes CUDA before/after pipeline;
total case includes preparation, inference, encode, collectives and validation,
excluding one-time model setup and final durable publication. Stats timing fields
describe the last successful case. Existing wall_seconds is retained unchanged.
Compare pipeline_infer_wall_seconds with 1573.101990563795; report all rank_memory
entries and speedup only after actual success. **No hybrid GPU benchmark has been
run by Codex; speedup and real PDD/CP/FSDP hook compatibility remain unverified.**

This is a new independent path based on `fdca08616314562e637b622359b45f6ad6817ac7`.
The A/B pilot, single-GPU PDD worker, Bernini, JoyAI, Visual and Audio/H3 production
code are unchanged. Only Video 1 is supplied; no Boogu, SAM, tracking, frame
selection, quality classifier, truncation or single-GPU fallback.

**Hybrid CP is not yet GPU-validated or production-ready.** The user confirmed
pure FSDP4 case05 success above; this does not validate CP, multi-job stability or
full-shard operation. Codex has not run these server commands.

## Source inspection and FSDP strategy

- Diffusers 0.40.0, commit `d035dcd7cc7c88e0a154609b62887d50bba9fdc2`:
  H3 denoiser calls `transformer_ref(...)` and inspects its forward signature;
  encoder calls `text_encoder.model(...)` directly.
- Alibaba PDD `335001fb9e5455d68a0caa18ec2e319072150328`:
  `apply_pdd_lora` mutates the module, adds FP32 updates and parallel heads,
  and registers a forward hook for its step arm.
- Reviewed ModelTC's FSDP2 layout at
  `02e26d591f7a04d5d1a074c9566d5dd4f22f6225` as a layout reference only.
  There is no dependency on its runtime or LightX2V inference.

Construction: Ref2VA from_pretrained (workflow only here) -> load_components
(no workflow) -> verify 12/3 shifts -> apply PDD once -> validate actual weights,
heads and 8-NFE plan -> FSDP2. Each forward retains the official PDD implementation,
8 evaluations / 9 grid points, CPU generator seed and native H3 timeline/geometry.

FSDP2 mutates modules in place, preserving signatures and PDD hooks. PDD's FP32
LoRA parameters and BF16 base linears use separate nested groups; no parameter
dtype is changed. Then refiner/block/root modules are sharded bottom-up.
This deliberately conservative grouping adds collective overhead; benchmark it
before making throughput claims.

H3's conditioner `text_encoder.model` and its decoder layers are also sharded.
Its unused outer LM head stays on CPU. The conditioner root explicitly reshards
after encoding. VAE/audio VAE and other non-sharded modules are replicated on each
local GPU. **No ComponentsManager CPU offload**, and no independent mover owns
`transformer_ref`. In the default pure FSDP mode activations are not sequence-parallel; two GPUs do not guarantee
every workload will fit.

Requires the inspected Diffusers 0.40.0 and an existing Torch stack providing
`torch.distributed.fsdp.fully_shard` (FSDP2). Missing/incompatible support is fatal;
the executor never upgrades dependencies or switches algorithms automatically.

## State and lifetime

`pair_id * pair_size : (pair_id + 1) * pair_size` is the fixed input row range.
Local indices modulo group-size assign independent Qwen workers to the group's
GPUs. Each lazily loads Qwen once, publishes every successful preparation
immediately and exits after scanning its partition, even with exhausted cases.
Then one torchrun session initializes NCCL/model/PDD once per rank and processes
all ready jobs sequentially. Pairs have no node-wide barrier.

Only rank zero schedules/broadcasts jobs, encodes videos and publishes manifests.
All ranks participate in the same transformer calls. Decode errors are exchanged
before any FSDP forward and do not require a reload. Encoding/validation failures
also leave the resident model intact. A failed model forward conservatively
restarts the entire distributed worker: FSDP hook/collective state may be partial.
An explicit stderr inference-error receipt plus a rank-zero in-flight journal
allows the supervisor to account for rank-asymmetric failure even if the
collective itself breaks. Pure NCCL/network, load/init failures or missing error
evidence stop the pair for operator diagnosis. User interruption does not consume
a case-failure budget.

```text
shard-000000/
  identity.json, pair.lock
  <source_index>-<row_digest>/
    original.mp4 -> original processed clip
    preparation/{descriptions,h3_prompt}.txt, prepared.json
    generation/raw.mp4, manifest.json
    attempts/{prepare,generate}_attempt_NNN.json
    failures/{prepare,generate}_attempt_NNN.json
    tmp/generate-NNN/raw.mp4
  sessions/<unique invocation>/config.json, qwen-*.log, h3-*.log, h3-*.json
```

- `manifest.json` exists: DONE, never overwritten or regenerated.
- Prepared marker only: skip Qwen; eligible for H3.
- Neither marker: eligible for Qwen, regardless of stale partial text/video.
- Text files are individually atomic; prepared marker is written last.
- Temporary video must decode and match count, 24 FPS, canvas and <= one-frame
  duration tolerance; fsync/rename raw video, then atomic manifest last.
- Temp files from interrupted attempts are ignored, not mistaken for completion.
  They are retained for diagnosis; no manual cleanup is required to resume.
- Summary is diagnostic only. Input row/config identity is immutable for a shard.
- Attempts are append-only. Default maximum is two **recorded failures** per
  stage; interrupted attempts remain eligible. `--retry-failed` grants exhausted
  cases one additional maximum-sized budget for that invocation. DONE stays DONE.
- Pair flock excludes concurrent executors. Linux parent-death signals bind
  children/ranks to launchers; inherited lock descriptors prevent overlapping
  work during teardown. SIGINT/SIGTERM forwards termination and escalates residual
  groups after a 60-second grace. Wait for cleanup before resuming.

Exit 0: scan completed; exit 2: cases exhausted their failure budgets; an
infrastructure error is nonzero. Other pairs continue independently in either case.

## Canonical server paths

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
export PAIR_PYTHON=/mnt/workspace/litengjie/data/person_replacement_deps/bernini-qwen-env/bin/python
export H3_PYTHON=/mnt/workspace/litengjie/data/person_replacement_deps/minimax-h3-env/bin/python
export H3_MODEL_ROOT=/mnt/workspace/public/pretrained/MiniMaxAI/MiniMax-H3
export H3_PDD_CODE_ROOT=/mnt/workspace/litengjie/data/vendor/MiniMax-H3-Acc-LoRAs
export PAIR_INPUT=/mnt/workspace/liutao/X_human_data/collected_face_2.jsonl
export PAIR_CLIPS=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped
# Retain the server's already-confirmed exact value. Codex cannot inspect that
# server environment and does not invent a weight location.
: "${H3_PDD_LORA:?Set the existing server MiniMax-H3-Ref2VA-Acc-8Step.safetensors path}"
export H3_PDD_LORA
export PAIR_ROOT=/mnt/workspace/litengjie/data/person_replacement/h3_pdd_pairs_v1
COMMON=(--input-jsonl "$PAIR_INPUT" --clips-root "$PAIR_CLIPS"
  --output-root "$PAIR_ROOT" --h3-python "$H3_PYTHON"
  --h3-model-root "$H3_MODEL_ROOT" --pdd-code-root "$H3_PDD_CODE_ROOT"
  --pdd-lora "$H3_PDD_LORA" --pair-size 1000 --seed 42 --resume)
PAIR_CLI=tools/person_replacement/run_h3_pdd_pair_executor.py
```

Caches are offline and redirected inside writable session directories:
HF_HOME/HF_HUB_CACHE, Torch, Triton, Inductor, XDG and TMPDIR. Read-only model,
vendor and dataset directories are never cache destinations. No dependency/model
downloads. The parent must be the Qwen environment; H3 runs in its own Python.

Import-only smoke in the isolated H3 environment (from the repository root;
no model loading or GPU work):

```bash
"$H3_PYTHON" - <<'PY'
from r2v_data_v2.person_replacement.h3_pair_generation import generate_loop
from r2v_data_v2.person_replacement.h3_pdd_distributed import *
print("H3 worker imports OK")
PY
```

Shard selection imports the parent manifest parser only when invoked by the
parent. Do not install `ijson` into the isolated H3 environment. On a torchrun
infrastructure failure, the executor prints the absolute worker log path and
its last 80 lines before raising; a missing/unreadable log does not mask the error.

Read-only scan/status (no model, child, lock, plan or artifact writes):

```bash
"$PAIR_PYTHON" "$PAIR_CLI" "${COMMON[@]}" --pair-id 0 --dry-run
"$PAIR_PYTHON" "$PAIR_CLI" "${COMMON[@]}" --pair-id 0 --status
```

After acceptance, first run and **identical command to resume**:

```bash
CUDA_VISIBLE_DEVICES=0,1 "$PAIR_PYTHON" "$PAIR_CLI" "${COMMON[@]}" --pair-id 0
# Explicit later retry of exhausted failures; completed cases are still skipped:
CUDA_VISIBLE_DEVICES=0,1 "$PAIR_PYTHON" "$PAIR_CLI" "${COMMON[@]}" --pair-id 0 --retry-failed
# Only after the following acceptance sequence succeeds:
"$PAIR_PYTHON" tools/person_replacement/run_h3_pdd_node.py "${COMMON[@]}" \
  --gpus 0,1,2,3,4,5,6,7 --pair-start 0
```

The node command defaults to four independent pairs, fixed non-overlapping ranges.
With eight GPUs, `--group-size 4` creates two groups (0–3, 4–7), while
`--group-size 8` creates one. Any distinct GPU list divisible by group-size is
allowed; each group keeps its original pair-id-based fixed input range.
The same node command resumes them. `--dry-run` or `--status` is forwarded
read-only to every pair. No cross-node work stealing or dynamic GPU selection.

## Mandatory acceptance sequence (operator only)

### Four-GPU resume of the existing failed case_05

Historical recipe for the formerly failed root. **The now-successful case05 is
DONE and must be skipped**; use the new-root hybrid benchmark recipe above for CP.

Use the **existing** CASE05_ROOT/CASE05_ARGS defined for the failed two-GPU run.
Do not change its input, seed, prompt, checkpoint, pair-id or pair-size. Inspect
the existing prepared.json: 311 frames, 1568x672, seed 42. Then:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
"$PAIR_PYTHON" "$PAIR_CLI" "${CASE05_ARGS[@]}" \
  --resume --group-size 4 --retry-failed --max-generate-attempts 1
```

This must skip Qwen, reuse prepared.json, and launch one four-rank torchrun.
Record wall_seconds, rank_memory for all four ranks, model_load_count,
pdd_apply_count and distributed_init_count. The latter three must remain 1 per
session (one initialization per rank). Four-GPU success/performance is **not yet
verified**; compare against two GPUs only after a valid output and manifest.
After successful acceptance, the node command with `--group-size 4` runs two
independent persistent groups on eight GPUs, each with four Qwen prepare workers.

Worker environments default `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
an explicitly supplied operator value is preserved. No dependency, PDD, prompt,
timeline, canvas, reference or FSDP wrapping policy changes accompany group sizing.

### Initial acceptance and interruption checks

1. Dry-run above.
2. Use a separate new acceptance root and the known one-row fixture:

   ```bash
   CASE05=artifacts/h3_pdd_two_person_8gpu_inputs/case_05.jsonl
   CASE05_ROOT=/mnt/workspace/litengjie/data/person_replacement/h3_pair_case05_v1
   CASE05_ARGS=(--input-jsonl "$CASE05" --clips-root "$PAIR_CLIPS" --output-root "$CASE05_ROOT"
     --h3-python "$H3_PYTHON" --h3-model-root "$H3_MODEL_ROOT"
     --pdd-code-root "$H3_PDD_CODE_ROOT" --pdd-lora "$H3_PDD_LORA"
     --pair-id 0 --pair-size 1 --seed 42 --resume)
   "$PAIR_PYTHON" "$PAIR_CLI" "${CASE05_ARGS[@]}" --dry-run
   CUDA_VISIBLE_DEVICES=0,1 "$PAIR_PYTHON" "$PAIR_CLI" "${CASE05_ARGS[@]}" --prepare-only
   ```

   Inspect prepared.json and compare with the historical case_05 job before
   inference: same source, exact prompt, seed 42, **311 frames, 1568x672**.
   If any differ, stop the acceptance comparison; do not lower frames/canvas
   or silently regenerate a different prompt to make it fit.
   Then run the same command without `--prepare-only`. Require valid raw.mp4
   and manifest.json; record wall_seconds and rank_memory. Memory counters are
   PyTorch peak allocated/reserved bytes, not full nvidia-smi process footprint.
3. Prepare an operator-verified three-row manifest in a new root: 311-, 124-,
   226-frame cases, in that order. Use `--pair-size 3 --prepare-only`, inspect
   artifacts, then resume without that flag. One h3-0.json must show
   model_load_count=1, pdd_apply_count=1, distributed_init_count=1,
   jobs_attempted=3, jobs_succeeded=3, jobs_failed=0, restart_required=false.
   Counts describe one logical distributed worker (one load/apply on each rank),
   not the sum of replicated rank operations. Compare per-case memory for growth.
4. New 6–10-case root: SIGTERM the parent during Qwen. Restart the identical
   command and verify prepared markers/text bytes unchanged and Qwen skips them.
5. Repeat during H3: completed manifests/raw bytes unchanged; interrupted case
   lacks manifest and reruns, later cases continue. Repeat with SIGKILL on Linux
   after checking descendant teardown/flock release.
6. Include one deliberately invalid media case in a private test manifest.
   Confirm durable failure history/budget exhaustion without blocking later cases.
7. Only then run the four-pair command. Do not point tests at production outputs.

Local regression evidence is recorded in the delivery report. All GPU memory,
throughput, real model kill/resume and four-pair acceptance results remain pending.
