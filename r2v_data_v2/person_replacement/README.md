# Bernini person-replacement pilot

Independent, sequential data generation; no changes to Visual/H3 pipelines,
no training, scheduler, structured Qwen output, or automatic quality judgement.
Only `collected_face_1.jsonl` is enabled. Source data and checkpoints are read-only.

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
makes exactly two plain-text calls: 16 uniformly sampled video frames for a short
source-person description, then text-only invention of a different ordinary
person. No JSON parsing, ethnicity/race inference, prompt enhancer or donor pool.
Both calls use at most 128 generated tokens. The prompt template is fixed in
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
resolver containment, lazy Qwen mock calls, read-only dry-run, output failure,
and a fake `torchrun` subprocess proving one requested case/seed is launched.
