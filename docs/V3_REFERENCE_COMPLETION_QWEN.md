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

The input is the existing completion JSONL manifest. This first Qwen version
accepts only:

- `reference_type == "subject"`;
- a non-null `completion_mask_path` containing a valid explicit binary mask.

`object` and `group` records fail during preflight. Directional masks remain
available to the frozen PowerPaint benchmark but are not accepted here. The
source must remain a binary-alpha RGBA PNG, and all existing source, context,
mask, canvas, and filesystem checks remain active.

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

## Whole-Image Editing And Mask Enforcement

Qwen-Image-Edit-2511 is not native mask-conditioned inpainting. The backend
receives exactly one expanded RGB completion canvas and may regenerate the
whole image. The completion mask is never passed to the Qwen pipeline. It is
used only by the shared post-generation hard compositing and acceptance checks.

After restoring model output to canvas size, publication applies these layers
in order:

1. restore every pixel outside the completion mask from the baseline canvas;
2. restore every visible source pixel exactly from the source RGBA RGB.

Only generated pixels inside the completion mask can survive. Visible entity
pixels and all pixels outside the mask must be byte-exact relative to their
respective sources. The source and explicit mask files must also retain their
original SHA-256 hashes. A candidate must be RGB, match the canvas size, change
the masked region, contain non-constant generated content there, and contain no
invalid values. A hard-check failure skips the judge.

## Structured Review

The benchmark reuses `QwenReferenceCompletionJudge` and
`ReferenceCompletionReview`; it does not introduce another review schema. The
judge receives the source entity on white, the completion canvas input, the
explicit mask, and the final hard-composited candidate. It must reject a new
person or second face, identity change, reconstructed occluder, implausible
body or clothing continuation, poor boundaries, or a background-only change
that does not improve reference usefulness. Acceptance requires every existing
review boolean to be true. Attractive appearance alone is insufficient.

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

## Artifacts

Each sample contains:

```text
<sample_id>/
  source_rgba.png
  source_white.png
  context_rgb.png                 # only when supplied
  baseline_canvas.png
  visible_mask.png
  completion_mask.png
  candidate_qwen_seed_0.png
  candidate_qwen_seed_17.png
  review_qwen_seed_0.json
  review_qwen_seed_17.json
  result.json
```

Each Qwen attempt in `result.json` records `backend`, `candidate_id`, `seed`,
the final full prompt and negative prompt, candidate path and hash, hard-check
report, judge status and verdict, review path, runtime, and reason. It does not
record PowerPaint `strategy` or `fitting_degree` fields.

The root `benchmark_summary.json` records the Qwen backend identifier,
processed/accepted/rejected counts, sorted hard-check rejection counts, and
sorted false judge-flag counts. It contains no image bytes or duplicated full
review payloads.

## Production Decision Boundary

This benchmark has no authority to publish or replace a production reference.
The original source-faithful RGBA remains canonical on every failure and after
every experimental run. Human review is mandatory even after the structured
judge accepts a candidate. Production integration, alpha extraction, SAM3
re-segmentation, automatic eligibility detection, and generated-reference
publication are intentionally out of scope and require a separate task.
