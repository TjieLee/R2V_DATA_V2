# V3 Qwen Reference Completion Benchmark

## Status And Goal

This is an independent offline benchmark for completing the same incomplete,
occluded, or truncated person with a local Qwen-Image-Edit-2511 model. It is
not a V3 production stage and is not registered in `run_pipeline_v3.py`. It
does not update `clip.json`, pairing, instructions, exports, selected
references, or any production dataset.

The production reference remains the source-faithful RGBA artifact. A Qwen
candidate must pass the shared hard checks, the structured Qwen completion
judge, and subsequent human review. A failed benchmark leaves the original
RGBA reference unchanged.

PowerPaint v2-1 remains frozen as a historical negative experiment. Qwen is a
separate benchmark, not a fallback or an automatic competitor. One successful
manual Qwen example shows only that the direction may be viable; it does not
establish production eligibility. A diverse multi-sample benchmark is still
required, and any production integration must be a separate future task.

## Local Model

The backend uses the local model directory:

```text
/mnt/workspace/public/pretrained/Qwen/Qwen-Image-Edit-2511
```

It loads `diffusers.QwenImageEditPlusPipeline` with
`local_files_only=True`, disables its progress bar, and moves the pipeline to
the configured device. It does not download a model, use a network model ID,
load Object-Remover LoRA, call a closed image API, or enable CPU offload.
`close()` releases the pipeline and clears the CUDA cache when CUDA is
available.

The default model-space short side is 1024 and both dimensions are rounded to
a multiple of 16 while preserving aspect ratio. Seeds run in deterministic
order: `0`, then `17`.

## Supported Input

The input is the existing completion JSONL manifest and `reference_type` must
remain `subject`; `object` and `group` fail during preflight. In the default
`whole_canvas` mode, `completion_mask_path` may be `null`. `completion_sides`
still determine canvas expansion, while `completion_start_ratio` is retained
for provenance but is not used. Source RGBA, optional context, canvas, and
filesystem checks remain active.

The historical `explicit_mask` comparison mode requires a non-null valid
binary `completion_mask_path` and preserves the original explicit-mask
validation. PowerPaint continues to support its directional and explicit mask
modes without any behavior change.

## Prompt Contract

The public default prompt is English:

> Complete the missing parts of the same single person shown in this image.
> Preserve the person's identity, face, hairstyle, clothing, accessories,
> pose, body proportions, lighting, perspective, and all visible appearance
> details. Extend only the incomplete body and clothing naturally into the
> empty region. Do not create another person, another face, duplicate body
> parts, new clothing, or new foreground objects. Do not alter the already
> visible person. Keep the background plain white.

The deterministic entity phrase is appended as `Entity description: ...` and
does not replace the core constraints. The default negative prompt rejects a
second or different person, duplicate faces or bodies, extra or disconnected
limbs, anatomy and identity changes, different clothing, new foreground
objects, text, watermarks, and logos. `--prompt` and `--negative-prompt` may
override these defaults, but blank values fail closed. There is no per-sample
LLM prompt rewrite.

## Compositing Modes

Qwen-Image-Edit-2511 is not native mask-conditioned inpainting. The backend
receives exactly one expanded RGB completion canvas and may regenerate the
whole image. No completion mask or editable region is passed to the Qwen
pipeline.

`whole_canvas` is the default. After restoring model output to canvas size,
code restores only the original visible entity pixels exactly from the source
RGBA. Every other canvas pixel remains Qwen output. This permits continuous
clothing, anatomy, a near-white background, and slight natural shadows. The
binary `editable_region` is exactly the inverse of the visible entity mask. It
is used only for hard-check statistics, judge visualization, and provenance;
it is not a model input mask and does not crop the candidate.

The whole-canvas hard checks require RGB canvas-sized output, exact visible
source pixels, unchanged source bytes, real change outside the visible region,
non-constant generated-region content, and valid pixel values. Background
complexity, a second person, identity or clothing changes, new objects or
scenes, watermarks, and structural continuity are left to the structured judge
and human review rather than an uncalibrated hard classifier.

`explicit_mask` remains only as a historical comparison and debugging mode. It
retains the previous two-layer compositing order:

1. restore every pixel outside the explicit mask from the baseline canvas;
2. restore every visible source pixel exactly from the source RGBA RGB.

This geometric postprocessing can create artificial vertical or diagonal cuts
and break clothing or anatomy when a whole-image editor generated a coherent
result outside the narrow polygon. It is therefore no longer the Qwen default.
Explicit mode still requires exact pixels outside the mask, a changed and
non-constant masked region, and unchanged source and mask hashes. Any hard-check
failure skips the judge in either mode.

## Structured Review

The benchmark reuses `QwenReferenceCompletionJudge` and
`ReferenceCompletionReview`; it does not introduce another review schema. The
judge receives the source entity on white, the baseline expanded canvas, the
mode-specific region image, and the final candidate. Image 3 is labeled
`completion mask` in explicit mode and
`editable region; this is not a model input mask` in whole-canvas mode.

For whole-canvas review, the judge is told that Qwen may edit the full canvas
and that code has already restored the visible person exactly. It must reject a
second person or face, identity or clothing changes, a new scene or salient
object, text or logos, unnatural body and clothing continuation, isolated
fragments, incomplete anatomy, and obvious vertical or diagonal crop
boundaries. White or near-white background and slight natural shadow are
allowed. Acceptance still requires every existing review boolean to be true;
attractive appearance alone is insufficient.

Both seed candidates and their reviews are retained. The earliest candidate
that passes hard checks and structured review is recorded for compatibility;
later candidates are still evaluated and preserved. There is no PowerPaint
fallback and no cross-backend quality selection.

## Execution

```bash
python -m tools.run_v3_reference_completion_qwen \
  --manifest /mnt/workspace/litengjie/data/reference_completion_qwen.jsonl \
  --benchmark-root /mnt/workspace/litengjie/data/reference_completion_qwen_benchmarks/run-001 \
  --judge-model /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
```

The benchmark root must be a new directory strictly below:

```text
/mnt/workspace/litengjie/data/reference_completion_qwen_benchmarks/
```

All records are preflighted before this root is created. Each sample is built
in a temporary sibling directory and atomically renamed into place. Exceptions
remove the temporary sample. The benchmark never writes to production run or
dataset roots, the public dataset, or `selected/`.

Use `--compositing-mode explicit_mask` only to reproduce the historical
comparison. The CLI default is `whole_canvas` and does not require a manifest
mask.

## Artifacts

Each default whole-canvas sample contains:

```text
<sample_id>/
  source_rgba.png
  source_white.png
  context_rgb.png                 # only when supplied
  baseline_canvas.png
  visible_mask.png
  editable_region.png
  candidate_qwen_seed_0.png
  candidate_qwen_seed_17.png
  review_qwen_seed_0.json
  review_qwen_seed_17.json
  result.json
```

Explicit mode publishes `completion_mask.png` instead of
`editable_region.png`.

Each Qwen attempt in `result.json` records `backend`, `candidate_id`, `seed`,
`compositing_mode`, the final full prompt and negative prompt, candidate path
and hash, hard-check report, judge status and verdict, review path, runtime, and
reason. The top-level result records `compositing_mode`. Whole-canvas results
set completion-mask mode/path/hash to `null` and record
`editable_region_path`; explicit results preserve completion-mask provenance
and set `editable_region_path` to `null`. Qwen results do not record PowerPaint
`strategy` or `fitting_degree` fields.

The root `benchmark_summary.json` records the Qwen backend identifier,
processed/accepted/rejected counts, sorted hard-check rejection counts, and
sorted false judge-flag counts. It also records deterministic
`compositing_mode_counts`. It contains no image bytes or duplicated full review
payloads.

## Prompt Language Comparison

The current English/Chinese prompt A/B examples were affected by narrow
explicit-polygon compositing, so they do not support a conclusion about prompt
language quality. A fair next comparison must use whole-canvas mode with the
same input, seed, and inference parameters, changing only the prompt. The
official default remains the English prompt above; a Chinese prompt is still
available only through an explicit CLI override. The successful manual Qwen
example is closer to whole-canvas behavior, but it does not establish
production readiness.

## Production Decision Boundary

This benchmark has no authority to publish or replace a production reference.
The original source-faithful RGBA remains canonical on every failure and after
every experimental run. Human review is mandatory even after the structured
judge accepts a candidate. Production integration, alpha extraction, SAM3
re-segmentation, automatic eligibility detection, and generated-reference
publication are intentionally out of scope and require a separate task.
