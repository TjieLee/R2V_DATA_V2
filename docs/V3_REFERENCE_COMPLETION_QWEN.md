# V3 Qwen Localized Entity Completion Benchmark

## Status And Scope

This is an independent offline benchmark for localized completion with the
local Qwen-Image-Edit-2511 model. It supports `subject`, `object`, and `group`
references. It repairs only a missing or incomplete local part inside the
reference's current canvas. It does not require a complete body, a complete new
object, or a larger group scene.

The benchmark is not a V3 production stage and is not registered in
`run_pipeline_v3.py`. It does not update `clip.json`, selected references,
pairing, instructions, exports, or any production dataset. The canonical
production reference remains the source-faithful RGBA image.

PowerPaint v2-1 remains frozen as a historical negative experiment and is not a
fallback. Its final evaluation and production rejection are unchanged.

## Local Model

The backend loads only:

```text
/mnt/workspace/public/pretrained/Qwen/Qwen-Image-Edit-2511
```

It uses `diffusers.QwenImageEditPlusPipeline` with `local_files_only=True`,
disables the progress bar, and moves the pipeline to the configured device. It
does not load Object-Remover LoRA, use a network model ID, call an external image
API, or use PowerPaint. Dtype compatibility supports either the current `dtype`
or legacy `torch_dtype` loader parameter.

## Modes

`localized_raw` is the default and the only active development mode. It supports
all three entity reference types:

- `subject`: repair a local missing part of the same person, animal, or subject
  without changing identity, appearance, clothing, pose, or outline.
- `object`: repair a local missing part of the same object while preserving
  geometry, material, texture, color, scale, and viewpoint.
- `group`: repair a local missing part of the same group while preserving its
  members, arrangement, relationships, appearance, scale, and viewpoint.

`whole_canvas` and `explicit_mask` remain available only as historical
experimental modes. Their artifact and compositing behavior is retained for
reproduction, but they are not defaults and are not being extended.

## Localized Input

`localized_raw` validates the source as an RGBA PNG with binary alpha and a
non-empty foreground. Code composites the visible source pixels over white at
the source's original width and height and publishes that input as
`input_source_white.png`.

The mode deliberately does not:

- expand or pad the canvas;
- use `completion_sides` or `completion_start_ratio`;
- read or apply `completion_mask_path`;
- build a completion mask or editable region;
- resize to `model_min_side` or force a multiple;
- pass `height`, `width`, a mask, or an editable region to Qwen;
- restore visible pixels after generation;
- perform mask compositing or alpha extraction.

The manifest continues to use `CompletionManifestRecord`; legacy geometry
fields are accepted only for compatibility and are ignored by `localized_raw`.

## Generation Prompts

The default English prompt is selected by `reference_type`. Subject completion
preserves identity, appearance, pose, proportions, clothing, texture, and other
visible characteristics. Object completion preserves geometry, material,
texture, color, scale, and viewpoint. Group completion preserves members,
arrangement, relationships, appearance, scale, and viewpoint.

All three prompts request only the missing local part, prohibit a new instance
or expanded composition, and keep the background plain and unchanged. The
entity phrase is recorded as metadata but is never appended to the generation
prompt.

The short Chinese reproduction prompt is:

```text
把这张图中残缺的部分补充完整，不要生成新的实例
```

It is an explicit CLI override and historical comparison baseline, not the
formal default. Any other `--prompt` value is a custom override applied to all
records. The official-like default negative prompt is exactly one space.

## Official-Like Call

The default localized call uses seed `0`, 40 inference steps,
`true_cfg_scale=4.0`, and `guidance_scale=1.0`:

```python
result = pipeline(
    image=[input_rgb],
    prompt=final_prompt,
    negative_prompt=" ",
    generator=torch.Generator(device=config.device).manual_seed(seed),
    true_cfg_scale=4.0,
    guidance_scale=1.0,
    num_inference_steps=40,
    num_images_per_prompt=1,
)
```

No `height`, `width`, mask, completion mask, or editable region is sent.

## Raw Candidate And Hard Checks

The first returned PIL image is converted to RGB and saved at its native output
size as:

```text
candidate_qwen_localized_seed_0.png
```

It is not resized, cropped, composited, or combined with restored source pixels.
`result.json` records both `input_size` and `output_size`; they are allowed to
differ.

Hard checks are deliberately entity-agnostic. They require a PIL image that can
be converted to RGB, positive dimensions, uint8 finite pixels, non-constant
content, a result different from the white input, unchanged source bytes,
reopenable published PNGs, and a recorded candidate SHA-256. They do not inspect
masks, visible-pixel equality, entity completeness, anatomy, object geometry, or
group semantics.

## Structured Review

`QwenLocalizedReferenceCompletionJudge` reuses
`ReferenceCompletionReview`, strict JSON schema output, `json_object` fallback,
structured repair, and fail-closed behavior. The judge receives only:

1. source entity on white;
2. localized raw candidate.

It does not receive a baseline duplicate, mask, or editable region. The system
prompt states that this is local repair rather than full-instance
reconstruction, so a candidate cannot be rejected merely for completing only a
small genuinely missing area.

The type-specific review guidance checks subject identity and body consistency,
object instance geometry/material continuity, or group membership and
arrangement. For `object` and `group`, `identity_preserved` means that the result
is still the same object or group rather than a new or redesigned instance.
Every review boolean must be true for an `accept` verdict.

The judge can reject automatically but cannot approve production use. A passing
hard check plus judge `accept` produces `manual_review_pending`; every other
outcome is `rejected`. `localized_raw` never emits `accepted`, and
`accepted_candidate` is always `null`.

## Artifacts And Metadata

Each localized sample contains only:

```text
<sample_id>/
  source_rgba.png
  input_source_white.png
  candidate_qwen_localized_seed_0.png
  review_qwen_localized_seed_0.json
  result.json
```

The result records backend, mode, source identity and hash, input and output
sizes, prompt and language, inference parameters, candidate path and hash, hard
checks, judge status and verdict, manual-review state, and runtime. It also
records that localized completion is enabled while full reconstruction, canvas
expansion, completion masks, visible-pixel restoration, forced dimensions, and
entity-phrase prompt appending are disabled.

The deterministic root summary records `reference_type_counts`,
`manual_review_pending`, rejected and accepted counts, hard-check rejection
reasons, and false judge flags. `accepted` is always zero for `localized_raw`.

## Execution

```bash
python -m tools.run_v3_reference_completion_qwen \
  --manifest /mnt/workspace/litengjie/data/reference_completion_qwen.jsonl \
  --benchmark-root /mnt/workspace/litengjie/data/reference_completion_qwen_benchmarks/run-001 \
  --judge-model /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct \
  --mode localized_raw \
  --seed 0
```

The benchmark root must be a new directory strictly below:

```text
/mnt/workspace/litengjie/data/reference_completion_qwen_benchmarks/
```

All records are preflighted before the root is created. Each sample is built in
a temporary sibling directory and atomically renamed into place. Exceptions
remove the temporary sample. The benchmark never writes to production run or
dataset roots, the public dataset, or `selected/`.

Legacy canvas, padding, mask-overlap, and model-size CLI options apply only when
`--mode whole_canvas` or `--mode explicit_mask` is selected.

## Production Decision Boundary

This benchmark has no authority to publish, replace, or automatically qualify a
production reference. Human review remains mandatory even after a structured
judge accepts a localized candidate. Production integration, SAM3,
re-segmentation, alpha extraction, automatic eligibility, and generated
reference publication remain out of scope.
