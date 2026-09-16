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

Dry-run checks JSONL records, existing contained MP4 paths and output placement,
and uses `ffprobe -count_frames` to count decoded frames and inspect rational FPS.
It prints source timelines and internal generation plans without loading models,
creating output directories, or generating intermediate videos. This Mac cannot verify
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
  No midpoint sampling or image resize. Generated output instead receives full
  decoded-frame-count / rational-FPS validation; this is not semantic QA.
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
the requested seed, frame count and integer FPS, and writes `bernini_launcher.sh`
inside the new case output.
It keeps the official inference command/options; unknown control flow, case/config
overrides or prompt enhancement fail closed. The original checkout is not edited.
This is not a general shell rewriter or a vendored inference implementation.

The old baseline inherited 81 frames / 16 FPS. The timeline adapter now overrides
only those temporal options and seed, preserving 40 inference steps, guidance,
omega, ViT CFG, resolution and prompt-enhancer behavior. GPU controls such as
`CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `ULYSSES`
remain operator/official-launcher settings. Manually inspect a server one-case
result before generating more data. No real Qwen/Bernini/GPU inference has been
validated by these Mac tests; server library and official-code-version compatibility
remain runtime prerequisites.

## Full visual timeline (pinned upstream inspection)

Code must be at **`e6c2cf11843470b08491745947abcba9c3c6b079`** (validated before
generation). Inspection of `bernini/cli.py`, `pipeline.py`,
`data/utils/video_utils.py` and the VAE transform found:

- CLI `--fps` is an integer. It controls VAE source sampling, sparse ViT sampling
  (`fps // 8`), and output encoding. It is not a separate FPS parameter to temporal
  position embeddings; sampled shapes/content affect the model's temporal input.
- `smart_video_nframes(... frame_factor=4, add_one=True)` computes
  `floor((decoded_count / input_fps * requested_fps) / 4) * 4 + 1`, minimum 5.
  It builds rounded linspace indices over the source, then truncates that index
  list to `floor(num_frames / 4) * 4 + 1`. Thus raising the output cap alone is
  insufficient: mismatched input/CLI FPS can skip/repeat real source frames.
- VAE temporal shape is `4n+1`; output length is additionally bounded by the
  source-derived latent length. Upstream is not a reliable automatic tail padder.

The adapter chooses the smallest `4n+1 >= source_count`, minimum 5, and nearest
integer source FPS. For 81@16 it remains 81@16; 137@25 stays 137@25;
138@25 becomes 141@25 with exactly three repeated final frames. Clips below five
frames need up to four pads. With equal input/CLI FPS and legal count, upstream
VAE indices are exactly `0..N-1`; official sparse ViT sampling remains unchanged.

If padding **or fractional FPS** requires conversion, case-local
`bernini_input.mp4` preserves every real frame in order, appends only repeated
last frames, and assigns integer internal timestamps. Fractional clips need this
even with zero padding. The original file and exact frame-zero reference remain
unchanged. Input encoding uses libx264 CRF 0; FFmpeg and ffprobe must be on PATH.

Full Bernini generation writes `replacement_raw.mp4`. After validating its
internal count/FPS, normalization trims only synthetic tail frames and restores
the exact source rational FPS (e.g. `24000/1001`, not permanently 24). No FPS
resampling filter, interpolation, motion smoothing or `-shortest` is used.
Necessary output encoding uses libx264 slow / CRF 12; when already aligned,
promotion is a rename without reencoding. Final decoded count must exactly equal
source count, FPS must match, and duration tolerance is one source frame.
Decoded presentation timestamps are checked for constant cadence (allowing one
time-base tick of quantization, capped at one tenth of a frame); irregular VFR
inputs fail explicitly rather than silently changing their time positions.

There is **no silent five-second truncation**, no acceleration change, and no
arbitrary-length one-shot guarantee. OOM, positional limits and generation failures
remain explicit failures; no hidden truncation or sliding-window fallback exists.
The 00209 H200 one-shot memory limit and visual quality still require real testing.

## Outputs and failure handling

Each case saves `reference_frame0.jpg`, `source_person.txt`,
`replacement_person.txt`, `bernini_prompt.txt`, `bernini_case.json`,
`replacement.mp4`, and, **only after successful generation/media validation**,
`manifest.json`. Extra execution files are `bernini_launcher.sh`, `bernini.log`,
and the case-local `runtime/` cache. Timeline intermediates are `bernini_input.mp4`
(when needed), `replacement_raw.mp4` (renamed when no normalization is needed),
and `timeline.log` for encoding commands/errors.

Manifest relationships are always:

- `input_video_path`: generated replacement video.
- `reference_image_path`: original target clip's exact frame zero.
- `target_video_path`: original resolved clip, never the original full video.

Manifest also records `source_*` count/FPS numerator/denominator/size/duration,
`bernini_internal_frame_count`, `bernini_internal_fps`, `bernini_pad_frames`,
and `raw_output_*` / `output_*` timeline measurements. Success is written only
after raw validation, normalization and final validation all pass.

## Exact 00209 A/B run (operator only; not executed by Codex)

Use the same compatible environment and GPU settings as the f79e88 baseline.
Set `CLIPS_ROOT` to the already confirmed processed root; do not guess it. Select
the unique original JSONL row by basename, preserving it unchanged in a new
operator-owned one-row manifest (exclusive creation prevents overwriting):

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
git switch feature/person-replacement-bernini-timeline-v1
git pull --ff-only origin feature/person-replacement-bernini-timeline-v1
mkdir -p /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts
python - <<'PY'
import json
from pathlib import Path
source = Path('/mnt/workspace/liutao/X_human_data/collected_face_1.jsonl')
destination = Path('/mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/bernini_timeline_00209_input.jsonl')
matches = []
with source.open() as stream:
    for line in stream:
        if line.strip() and Path(json.loads(line)['video_path']).name == 'v_b597a0641cf76c22e880_00209.mp4':
            matches.append(line.rstrip('\n'))
if len(matches) != 1:
    raise ValueError(f'Expected one exact 00209 row; found {len(matches)}')
with destination.open('x') as stream:
    stream.write(matches[0] + '\n')
PY

python tools/person_replacement/run_bernini_pilot.py \
  --input-jsonl /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/bernini_timeline_00209_input.jsonl \
  --clips-root "${CLIPS_ROOT:?Set the confirmed processed-clip root}" \
  --bernini-code-root /mnt/workspace/litengjie/data/vendor/Bernini \
  --output-root /mnt/workspace/litengjie/data/R2V_DATA_V2/artifacts/bernini_timeline_00209_v1 \
  --limit 1 --seed 42 --dry-run
```

After checking the printed source/internal timeline, repeat the final command
**without `--dry-run`**, adding these explicit baseline model arguments:

```text
--qwen-model /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct
--bernini-config /mnt/workspace/public/pretrained/Bernini-Diffusers
```

Compare the new manifest's source/output count, rational FPS and duration, then
manually A/B replacement appearance and motion against the existing f79e88 output.
No overwrite is used; do not delete the old baseline or reuse an existing case dir.

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
Timeline tests cover legal-length planning, tail-only padding/trimming, fractional
FPS, strict launcher edits, final mismatch rejection and manifest gating. An
additional real codec test runs when FFmpeg/ffprobe are installed, otherwise skips;
mock tests alone do not prove a server's codec/model integration.
