# Independent accelerated H3 Ref2VA replacement candidates

This experiment does not replace Bernini or write production training tuples.
It only adds a separate CLI, deterministic prompt, subprocess adapters and tests.
No existing Audio/H3, Visual, Bernini, JoyAI, Qwen or shared dependency file changes.

## Implementation and validation plan

- [x] Inspect pinned upstream CLI, Ref2VA video loading, resolution and PDD head contracts.
- [x] Test six-section prompt, native timeline/range/aspect validation, and shared preparation.
- [x] Implement LightX2V subprocess and isolated PDD worker using official `apply_pdd_lora`.
- [x] Test checkpoint rejection, exact calls, Qwen release and per-backend publication gating.
- [x] Run focused tests, Ruff, compileall and diff check; inspect baseline diff before push.
- [ ] Operator H200 smoke/A-B: real weights, codec, memory, time and visual quality.

## Inspected sources (2026-09-17)

| Source | Pinned revision | Contract |
| --- | --- | --- |
| [ModelTC/Minimax-H3-Turbo](https://github.com/ModelTC/Minimax-H3-Turbo/tree/02e26d591f7a04d5d1a074c9566d5dd4f22f6225) | `02e26d591f7a04d5d1a074c9566d5dd4f22f6225` | `inference_minimax_h3.py`, `minimax_h3_ref2va_pipeline.py`, `resolution_util.py` |
| [Alibaba PAI](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs/tree/335001fb9e5455d68a0caa18ec2e319072150328) | `335001fb9e5455d68a0caa18ec2e319072150328` | `predict_ref2v.py`, `minimax_h3_pdd.py` |
| [LightX2V weight inventory](https://huggingface.co/lightx2v/Minimax-h3-Turbo/tree/3ec17a324ced54151364f24f8b5fb6bf7e26414f) | `3ec17a324ced54151364f24f8b5fb6bf7e26414f` | Confirmed exact requested Ref2VA 8-step Diffusers filename exists; no weights downloaded by Codex |
| [Diffusers v0.40.0](https://github.com/huggingface/diffusers/tree/d035dcd7cc7c88e0a154609b62887d50bba9fdc2/src/diffusers/modular_pipelines/minimax_h3) | `d035dcd7cc7c88e0a154609b62887d50bba9fdc2` | Video-reference rates, 24 FPS, `17n+5`, generation range 5–15 s |

Runtime validates upstream git HEAD and rejects edits to the required source files.
Both backends require the same **local** `MiniMaxAI/MiniMax-H3` root with
`transformer_ref`; there is no guessed server checkpoint path or automatic download.

### A: LightX2V

Exact checkpoint: `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`.
Diffusers format only, not FL2VA, 4-step or ComfyUI.
The official CLI receives 8 NFE, video/audio shift 6/3, alpha 8, scale 1,
reference-resize-mode `match`, megapixels 1.0, original video reference and the
explicit planned frame count. It adds the terminal sigma point internally.
No inference implementation is copied into this repository.

`H3_FSDP2=1` uses the H3 Python's `torch.distributed.run` and requires explicit
`H3_NPROC_PER_NODE`; it never hardcodes the visible GPU count. The pinned upstream
distributes jobs and pads a one-job batch across ranks; these extra identical jobs
are an upstream FSDP execution cost. Only rank 0's original-job output is selected,
by exact filename, and renamed to `raw.mp4` without transcoding. Other native
upstream outputs remain diagnostic files. Single process uses official CPU offload.

### B: Alibaba PDD

Exact checkpoint: `MiniMax-H3-Ref2VA-Acc-8Step.safetensors`.
This contains rank-64 backbone updates **and step-specific output heads**, not
ordinary PEFT LoRA weights. The worker checks for the native heads/updates before
loading and calls only official `apply_pdd_lora` at strength 1.0. It reads NFE,
head count and block size from the attached official controller/head objects.
8 NFE means `num_inference_steps=9`. Video/audio schedulers retain 12/3;
there is no LightX2V shift override, generic LoRA loader or reimplemented PDD math.
The official defaults are grid 32 / block 4; incompatible non-8-NFE plans fail.

Construction follows official `predict_ref2v.py` with one change in media modality:
`MiniMaxH3VideoReference.from_file(original_processed_clip)`, never frame zero.
PDD uses the official **single-process CPU-offload** route; no unverified PDD FSDP
implementation is added. `H3_FSDP2` controls LightX2V only. Comparing four-GPU
LightX2V and single-GPU/offloaded PDD wall time is not an equal-resource throughput
benchmark; both execution modes are reported.

## Timeline and framing

Source must be one whole processed MP4, 2–15 s, with a supported aspect ratio
within 1% (`21:9`, `16:9`, `4:3`, `1:1`, or transposes). No source crop/truncation.
Use the existing read-only timeline inspector (its CFR-cadence validation applies).

ModelTC computes `N=max(5, round(duration*24)); N += (5-N%17)%17`.
Its formula alone does **not** enforce its runtime's minimum duration. We explicitly
raise short outputs to the first legal >=5 s count, 124, and record drift. If an
aligned result exceeds 15 s, fail rather than shorten it: e.g. a 15.0 s source
rounds to 362 frames and cannot be handled by this first pilot. Requested source
duration is still recorded unchanged. A 138-frame 25 FPS source requests 5.52 s
and plans 141 frames at 24 FPS (5.875 s).

Output is kept at **native 24 FPS**. No final retime, interpolation, padding, frame
drop or source-exact normalization is performed by this adapter. Upstream H3 itself
resamples video conditioning to its native rate; this is not a frame-exact edit
guarantee. Every manifest says `production_timeline_compatible=false`.

Both adapters use ModelTC's 1.0 MP output geometry: 16:9 is **1376×768**, not the
PDD example's 1344×768. Other ratios follow the same 32-pixel rounding. The
`match` option concerns upstream **image** references; it does not replace H3's
stock video-reference preprocessing. These are qualitative native H3 candidates.

## Shared preparation, prompt and artifacts

`--backend both` is the fair comparison path: one `describe(original video)`, one
`invent(description)`, one deterministic prompt, then `Qwen.close()`, LightX2V,
PDD. Both receive the same prompt, original source, seed, requested duration,
native frame count, geometry and local base model. No Context-IR API, donor image,
segmentation, additional MLLM calls or audio QA.

For exact cross-model A/B, `--descriptions-from <existing Bernini manifest.json>`
reuses its two Qwen-produced descriptions without new Qwen calls. It validates
the manifest's original target path against the selected processed clip and
records the read-only provenance. Without this option the existing Qwen flow is
unchanged. No prior manifest or image is rewritten.

The template in `h3_prompt.py` always uses the official order:
`subject_definitions`, `summary`, `retention_analysis`, `detailed_description`,
`overall_soundscape`, `non_diegetic_music`. The original person in `<Video 1>` is
replaced by the invented appearance while pose, contacts, expression, gaze, camera,
background and timing are explicitly retained. The final two fields are `N/A`.

Each case keeps frame-zero provenance, both descriptions, prompt and source
timeline. Each backend has its own job, raw video, log and manifest. Success
requires a readable output with planned count, 24 FPS, native duration and shared
resolution. A runtime failure writes `failure.json`, never a success manifest,
and the other backend still runs. CLI returns nonzero if either backend failed.
No existing case is overwritten. For retries use a new output root.

`backend_wall_seconds` includes startup, weight loading, generation and encoding;
it is not isolated denoising time. Record warm/cold state when comparing speeds.

## Operator-only setup (Codex did not execute these commands)

Keep the Qwen-capable parent environment from the Bernini pilot unchanged. Create
a NEW H3 environment only. Select a PyTorch CUDA wheel index compatible with the
server driver; do not replace the main/Bernini environments. The following is a
setup recipe, not an H200-validated environment lock:

```bash
export H3_ENV=/mnt/workspace/litengjie/data/person_replacement_deps/minimax-h3-env
export HF_HOME=/mnt/workspace/litengjie/data/cache/h3/huggingface
export TORCH_HOME=/mnt/workspace/litengjie/data/cache/h3/torch
export XDG_CACHE_HOME=/mnt/workspace/litengjie/data/cache/h3/xdg
export TMPDIR=/mnt/workspace/litengjie/data/tmp/h3
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR"
test ! -e "$H3_ENV" && python3.12 -m venv "$H3_ENV"
export H3_PYTHON="$H3_ENV/bin/python"
"$H3_PYTHON" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url "${H3_TORCH_INDEX_URL:?Set the server-compatible PyTorch CUDA wheel index}"
"$H3_PYTHON" -m pip install diffusers==0.40.0 'transformers>=4.57,<6' \
  accelerate peft safetensors av pillow numpy sentencepiece protobuf

export H3_TURBO_CODE_ROOT=/mnt/workspace/litengjie/data/vendor/Minimax-H3-Turbo
export H3_PDD_CODE_ROOT=/mnt/workspace/litengjie/data/vendor/MiniMax-H3-Acc-LoRAs
git clone https://github.com/ModelTC/Minimax-H3-Turbo.git "$H3_TURBO_CODE_ROOT"
git -C "$H3_TURBO_CODE_ROOT" checkout --detach 02e26d591f7a04d5d1a074c9566d5dd4f22f6225
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs "$H3_PDD_CODE_ROOT"
GIT_LFS_SKIP_SMUDGE=1 git -C "$H3_PDD_CODE_ROOT" checkout --detach 335001fb9e5455d68a0caa18ec2e319072150328
```

Code clone destinations must be new; do not overwrite an existing vendor checkout.
Provision the base model and exact weights separately, then explicitly set:

```bash
: "${H3_MODEL_ROOT:?Set existing local MiniMaxAI/MiniMax-H3 root}"
: "${H3_LIGHTX2V_LORA:?Set local minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors}"
: "${H3_PDD_LORA:?Set local MiniMax-H3-Ref2VA-Acc-8Step.safetensors}"
export H3_MODEL_ROOT H3_LIGHTX2V_LORA H3_PDD_LORA
```

There are no weight downloads in the pilot. Public pretrained/data trees stay
read-only. Subprocesses enforce HF offline modes and redirect HF/Torch/Triton/
Inductor/temp caches to backend-local `runtime/`; no vendor writes or bytecode.
FFmpeg and ffprobe must be on the parent PATH for timeline validation.

## Exact 00209 commands

From the repo, using the unchanged **Qwen-capable parent Python**, reuse the
one-row manifest prepared for the Bernini timeline A/B. Verify its basename
before any inference; do not select the first arbitrary row of the full dataset:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
export H3_00209_JSONL=/mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/bernini_timeline_00209_input.jsonl
: "${BERNINI_00209_MANIFEST:?Set the existing successful 00209 Bernini manifest.json}"
export BERNINI_00209_MANIFEST
python - <<'PY'
import json, os
from pathlib import Path
rows = [json.loads(x) for x in Path(os.environ['H3_00209_JSONL']).read_text().splitlines() if x.strip()]
assert len(rows) == 1
assert Path(rows[0]['video_path']).name == 'v_b597a0641cf76c22e880_00209.mp4'
PY
```

If that one-row manifest does not exist, use the existing Bernini README's
exclusive-create basename selection recipe; do not modify the public JSONL.
`CLIPS_ROOT` must remain the previously confirmed processed-clip root.

Dry-run (no model, worker or output-directory creation):

```bash
python tools/person_replacement/run_h3_ref2va_pilot.py \
  --input-jsonl "$H3_00209_JSONL" --clips-root "${CLIPS_ROOT:?}" \
  --descriptions-from "$BERNINI_00209_MANIFEST" \
  --output-root /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/h3_00209_both_v1 \
  --backend both --limit 1 --seed 42 --dry-run
```

Recommended shared-description/prompt A/B (sequential backends):

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 H3_FSDP2=1 H3_NPROC_PER_NODE=4 \
python tools/person_replacement/run_h3_ref2va_pilot.py \
  --input-jsonl "$H3_00209_JSONL" --clips-root "${CLIPS_ROOT:?}" \
  --descriptions-from "$BERNINI_00209_MANIFEST" \
  --output-root /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/h3_00209_both_v1 \
  --backend both --limit 1 --seed 42
```

Separate backend debugging commands (new outputs; the same validated existing
Bernini descriptions keep preparation identical between these runs):

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 H3_FSDP2=1 H3_NPROC_PER_NODE=4 \
python tools/person_replacement/run_h3_ref2va_pilot.py \
  --input-jsonl "$H3_00209_JSONL" --clips-root "${CLIPS_ROOT:?}" \
  --descriptions-from "$BERNINI_00209_MANIFEST" \
  --output-root /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/h3_00209_lightx2v_v1 \
  --backend lightx2v --limit 1 --seed 42

CUDA_VISIBLE_DEVICES=2 H3_FSDP2=0 H3_NPROC_PER_NODE=1 \
python tools/person_replacement/run_h3_ref2va_pilot.py \
  --input-jsonl "$H3_00209_JSONL" --clips-root "${CLIPS_ROOT:?}" \
  --descriptions-from "$BERNINI_00209_MANIFEST" \
  --output-root /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/h3_00209_pdd_v1 \
  --backend pdd --limit 1 --seed 42
```

H200 validation remains necessary for environment compatibility, LoRA tensor
compatibility, FSDP/NCCL, peak VRAM/RAM, codec output, runtime and motion/identity
quality. Without `--descriptions-from`, descriptions shared with an older Bernini
run are not guaranteed by rerunning Qwen; compare the saved text files before
claiming identity-matched A/B. No acceleration speedup or production readiness is claimed.

## Local verification

```bash
.venv/bin/python -m pytest tests/person_replacement -q
.venv/bin/ruff check r2v_data_v2/person_replacement/h3_*.py \
  tools/person_replacement/*h3*.py tests/person_replacement/test_h3*.py
.venv/bin/python -m compileall r2v_data_v2/person_replacement tools/person_replacement
git diff --check
```

Local result: **80 passed, 4 skipped**. The four existing Bernini real-codec
tests require FFmpeg/ffprobe, absent on this Mac. Model contracts use stubs, not
real weights. Pure resolution calculations were additionally checked against
the pinned ModelTC helper for all seven supported ratios. Independent read-only
code review approved after adding full PDD parameter coverage and preflight
failure isolation. No server/model/GPU experiment was executed.
