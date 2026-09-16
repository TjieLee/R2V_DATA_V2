# Person-replacement pilots: Bernini baseline and JoyAI

Independent, sequential data generation; no changes to Visual/H3 pipelines,
no training, scheduler, structured Qwen output, or automatic quality judgement.
Only `collected_face_1.jsonl` is enabled. Source data and checkpoints are read-only.

**Bernini is the unchanged legacy qualitative baseline:** its upstream fixed
81-frame launcher is **not full-timeline compliant for long clips**. JoyAI below
is the first backend implementing the full-source timeline contract. This is a
tested adapter contract, not a claim of verified real GPU inference.

## First server check (no models, no output writes)

From the repository root, first set `CLIPS_ROOT` to the **confirmed processed-clip
root for this JSONL**. It is intentionally required rather than guessed from
`source_video_path` or from another dataset's defaults.

```bash
python tools/person_replacement/run_bernini_pilot.py \
  --input-jsonl /mnt/workspace/liutao/X_human_data/collected_face_1.jsonl \
  --clips-root "${CLIPS_ROOT:?Set the confirmed processed-clip root first}" \
  --output-root /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/person_replacement_bernini_pilot \
  --limit 10 --seed 42 --dry-run
```

Dry-run checks JSONL records, existing contained MP4 paths and output placement;
it does not decode videos, inspect checkpoints, load models or invoke Bernini.
It prints selected cases and output directories. This Mac checkout cannot verify
the server's input rows or their clip root.

After checking those paths, start with one case in a **new output directory**:

```bash
python tools/person_replacement/run_bernini_pilot.py \
  --clips-root "${CLIPS_ROOT:?Set the confirmed processed-clip root first}" \
  --bernini-code-root "${BERNINI_CODE_ROOT:?Set the official Bernini code checkout}" \
  --output-root /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/person_replacement_bernini_one \
  --qwen-model /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct \
  --bernini-config /mnt/workspace/public/pretrained/Bernini-Diffusers \
  --limit 1 --seed 42
```

`BERNINI_CODE_ROOT` is also the default for `--bernini-code-root`. It must contain
`scripts/bernini/run_v2v.sh` and `infer_multi_gpu.py`; the checkpoint directory is
**not** assumed to contain inference source. Neither is copied into this repo.

## Existing utilities reused

- `v3.production_source.JeaVideoMotionAdapter.resolve_clip_path`: explicit-root
  resolver with file existence, containment and symlink-escape checks. Only the
  processed `video_path` is resolved. `source_video_path` remains provenance and
  is never model input or the frame source.
- `v3.frames.OpenCvFrameDecoder.decode_indices(video, [0])`: exact decoded frame
  zero from the resolved target clip, saved as `reference_frame0.jpg` (JPEG 95).
  No midpoint sampling or image resize. The same decoder checks that generated
  output has a readable first frame; this is not semantic QA.
- `manifest.iter_source_records`, `naming.clip_uid`,
  `reconciliation.write_json_atomic`: JSONL reading, stable naming, JSON writing.

## Model/runtime contract

Qwen lazily loads one local `Qwen3VLForConditionalGeneration` and `AutoProcessor`
with `local_files_only=True`, `device_map="auto"`, `dtype="auto"`. Per case it
makes exactly two plain-text calls: the source video is sampled using the
Qwen3-VL processor's official default video policy (currently 2 FPS) for a short
source-person description, then a text-only call invents a different ordinary
person. The pilot does not override `fps` or `num_frames`. No JSON parsing,
ethnicity/race inference, prompt enhancer or donor pool is used. Both calls use
at most 128 generated tokens. The prompt template is fixed in
`pipeline.build_prompt`.

Use an isolated Qwen3-VL-capable Transformers/PyTorch/Accelerate environment with
its video decoding dependencies (e.g. PyAV). No shared requirements are changed.
All selected cases are described first; Qwen is released before launching Bernini
to avoid competing GPU residency. The subprocess uses `torchrun` from `PATH`,
which must belong to a compatible official Bernini environment. It runs with the
official checkout as its working directory. Hugging Face offline flags and
case-local caches are set; missing local dependencies/checkpoints must be installed
or provisioned separately by the operator.

Official interfaces inspected on 2026-09-16:

- [v2v case](https://github.com/bytedance/Bernini/blob/main/assets/testcases/v2v/v2v_case1.json):
  `task_type`, `prompt`, `video`, `output`, serialized only by `serialize_case()`.
- [launcher](https://github.com/bytedance/Bernini/blob/main/scripts/bernini/run_v2v.sh):
  seed is a command-line argument, not a case field.
- [Qwen3-VL Transformers API](https://huggingface.co/docs/transformers/model_doc/qwen3_vl).

**Upstream launcher caveat:** the inspected script overrides `CASE_PATH` with a
loop over three bundled examples and hardcodes seed 42. The thin adapter reads
the operator's launcher and removes only that exact known demo loop, substitutes
the requested seed, and writes `bernini_launcher.sh` inside the new case output.
It keeps the official inference command/options; unknown control flow, case/config
overrides or prompt enhancement fail closed. The original checkout is not edited.
This is not a general shell rewriter or a vendored inference implementation.

The inspected official defaults are 8 processes, 81 frames, 16 FPS; this pilot
does not override them or assert full-clip duration/frame alignment with the
original. GPU controls such as `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `ULYSSES`
remain operator/official-launcher settings. Manually inspect a server one-case
result before generating more data. No real Qwen/Bernini/GPU inference has been
validated by these Mac tests; server library and official-code-version compatibility
remain runtime prerequisites.

## Outputs and failure handling

Each case saves `reference_frame0.jpg`, `source_person.txt`,
`replacement_person.txt`, `bernini_prompt.txt`, `bernini_case.json`,
`replacement.mp4`, and, **only after successful generation/media validation**,
`manifest.json`. Extra execution files are `bernini_launcher.sh`, `bernini.log`,
and the case-local `runtime/` cache.

Manifest relationships are always:

- `input_video_path`: generated replacement video.
- `reference_image_path`: original target clip's exact frame zero.
- `target_video_path`: original resolved clip, never the original full video.

Existing case directories are not overwritten. A failure retains diagnostics
but does not create a success manifest; this minimal pilot has no resume machinery.
Use a new output root for a retry. Keep server outputs below
`/mnt/workspace/litengjie/data`; no writes to public dataset/pretrained or
`X_human_data` trees are permitted.

## Local tests (no weights or server data)

```bash
.venv/bin/python -m pytest tests/person_replacement -q
```

Tests cover prompt/case/manifest contracts, frame-zero extraction calls, real
resolver containment, lazy Qwen mock calls including preservation of Qwen's
default video sampling policy, read-only dry-run, output failure, and a fake
`torchrun` subprocess proving one requested case/seed is launched.

## JoyAI full-timeline pilot

The official [JoyAI-Video-Edit checkout](https://github.com/jd-opensource/JoyAI-Video-Edit/tree/69fdaea89e2fd7708435790d73eb30989e722a5f)
must be exactly `69fdaea89e2fd7708435790d73eb30989e722a5f`, with clean
`deploy/xvideo` sources. It is external, not vendored. This pilot uses direct
text-only V2V, `ref_image=None`, and **no PromptEnhancer, WebSocket/FastAPI server,
donor image, semantic QA or external API**.

### External prerequisites and read-only models

Run the top-level pilot with the existing Qwen-capable Python. Supply a separate
JoyAI environment's executable via `--joyai-python` / `JOYAI_PYTHON`, and the
pinned checkout via `--joyai-code-root` / `JOYAI_CODE_ROOT`. Neither is guessed.
Install/provision JoyAI's own dependencies separately; this patch changes no
shared requirements and installs/downloads nothing. The worker needs OpenCV,
NumPy and Pillow as well as the official JoyAI dependencies. It imports only the
lightweight timeline helper from this repo, not the Visual/H3 pipeline.

Unified `--joyai-model-root` / `JOYAI_MODEL_ROOT` defaults to the server-confirmed
read-only `/mnt/workspace/public/pretrained/jdopensource/JoyAI-Video-Edit`.
`validate()` requires:

```text
JoyAI-Video-Edit/
  dit/joyai_video_edit_dit_0811.pth
  vae/config.json
  vae/diffusion_pytorch_model.safetensors
```

DiT and VAE paths are derived from this one root. Missing **0811** is fatal;
there is no silent **0804** fallback and no separate DiT/VAE flags to maintain.

`--joyai-text-encoder-ckpt` / `JOYAI_TEXT_ENCODER_CKPT` defaults to the separately
confirmed `/mnt/workspace/public/pretrained/MiMo/MiMo-VL-7B-RL-2508`. MiMo is not
assumed to live inside JoyAI's root, and never falls back to SFT. Validation checks
the local config, tokenizer/processor files and single safetensors or every shard
listed in `model.safetensors.index.json`. It does not load tensors or hash weights.

All these directories remain read-only. Parent Qwen caches use the new case's
`qwen_runtime/`; JoyAI uses `joyai_runtime/` as cwd and for Hugging Face, Torch,
extensions, Inductor, Triton, CUDA and temporary caches. Inherited cache locations
are overridden, network model lookup is disabled, and child bytecode writes are
disabled. Server outputs remain below `/mnt/workspace/litengjie/data`.

### First server command: dry-run

Confirm the actual `CLIPS_ROOT` for `collected_face_1.jsonl` first, then run from
the R2V repository root:

```bash
python tools/person_replacement/run_joyai_pilot.py \
  --clips-root "${CLIPS_ROOT:?Set the confirmed processed-clip root}" \
  --output-root /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/person_replacement_joyai_check \
  --limit 1 --seed 42 --dry-run
```

This validates input/resolved paths, decodes the selected source to count **all**
frames and prints timeline information. It loads no model, starts no subprocess,
and creates no output case directory. Full decoding rather than trusting container
frame-count metadata catches truncated inputs; it can take time for long clips.

After inspecting dry-run paths, the operator can start one real case:

```bash
python tools/person_replacement/run_joyai_pilot.py \
  --clips-root "${CLIPS_ROOT:?Set the confirmed processed-clip root}" \
  --joyai-code-root "${JOYAI_CODE_ROOT:?Set the pinned JoyAI checkout}" \
  --joyai-python "${JOYAI_PYTHON:?Set the isolated JoyAI Python executable}" \
  --output-root /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/person_replacement_joyai_one \
  --limit 1 --seed 42
```

The two confirmed model roots are CLI defaults; they need not be repeated.
The parent completes the existing two plain-text Qwen calls per selected case,
then calls `qwen.close()` before any JoyAI subprocess. `joyai_prompt.txt` uses a
separate positive replacement/preservation template, not Bernini's wrapper.
The existing Qwen prompts/default video sampling remain unchanged.

### Exact runtime and timeline contract

The adapter uses `JoyOmniRuntime.load(dit_ckpt, vae_ckpt=..., text_encoder_ckpt=...)`,
`StreamingSettings(num_inference_steps=2, seed=...)`, and
`create_v2v_session(prompt, settings=..., ref_image=None)` from the pinned
`deploy/xvideo/serving/joyomni_streaming.py`.

1. OpenCV obtains source FPS and decodes every frame in order; actual decoded
   count must agree with the container count when one is reported. Width/height
   and visual duration (`count / fps`) are recorded before generation.
2. The worker decodes the entire source, with no fixed seconds/frame cap, and
   calls `push_frame()` once per source frame with a consecutive sequence ID.
   It drains each submitted chunk with `wait_async_result()` to bound memory.
3. `flush_pending()` performs the **official** repeat-last-frame tail padding.
   Only `valid_count` real output frames are written. Missing, duplicate or
   reordered sequence IDs fail closed; arbitrary excess frames are not silently
   cropped. The session closes on success and failure.
4. The official `runtime.lossless_output=True` transport returns RGB arrays,
   avoiding an intermediate JPEG step. OpenCV writes MP4 (`mp4v`) at source FPS.
   This is not a promise of lossless final MP4 compression.
5. Worker and parent both verify decoded output count equals source count,
   FPS matches within floating tolerance, and duration differs by at most one
   source frame. This is a visual count/FPS contract, not preservation of audio
   or per-frame variable-rate timestamps.

Resolution uses official `generate_video_image_bucket()` at the runtime's
720×1248 area scale, selecting the closest aspect ratio with matching orientation
and VAE alignment. The official session resizes the **whole frame**; no center crop
is added. Discrete buckets can slightly change aspect ratio; source and output
dimensions are recorded and pixel-identical resolution is not promised.

Success writes `manifest.json` only after subprocess success and timeline checks.
It includes backend/pinned-code/checkpoint/seed provenance, source/output frame
count, FPS, dimensions and duration, both person descriptions and the JoyAI prompt.
`reference_frame0.jpg` still comes from exact original target frame 0, before any
JoyAI resize. No original video is copied. Failure leaves logs/request/artifacts
but no success manifest; no overwrite or resume machinery is introduced.

### Validation boundary

Local tests use mock runtimes, fake external executables and tiny synthetic MP4s:
137 frames (>5 seconds), 138 frames with a padded tail, 23.976/25 FPS, ordering,
missing model files, stale code commit, timeline mismatch, manifest withholding,
read-only dry-run and Qwen release order. Existing Bernini tests remain intact.

H200 is the intended server target. **No real JoyAI/Qwen/GPU job was run here and
no speed or quality result is claimed.** The server one-case pilot must still
validate environment/checkpoint compatibility, GPU memory, streaming behavior,
person replacement fidelity and preservation of scene/action.
