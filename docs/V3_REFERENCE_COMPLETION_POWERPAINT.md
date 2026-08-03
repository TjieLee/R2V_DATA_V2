# V3 PowerPaint Reference Completion Benchmark

## Status And Scope

This is an offline PowerPaint v2-1 experiment. It is not a production V3
pipeline stage and is not included in `run_pipeline_v3.py`. It does not update
`clip.json`, pairing, instructions, exports, or selected references.

The benchmark tests whether a truncated, occluded, or local entity reference can
be extended into a single coherent entity on a larger RGB canvas. This differs
from the [background contextualization benchmark](V3_REFERENCE_INPAINT_BENCHMARK.md),
which generates only transparent background around a complete source-faithful
RGBA entity and never completes the entity itself.

The production reference remains the source-faithful RGBA artifact. PowerPaint
does not guarantee identity preservation, so every candidate needs the
structured Qwen review described below and subsequent human inspection. A
rejected run keeps the original reference; it never promotes an unreviewed
candidate and never falls back to a background-contextualization output.

## External PowerPaint V2-1 Layout

The backend uses the official custom BrushNet classes from a local checkout. It
does not use the generic Hugging Face `DiffusionPipeline` and does not download
missing assets.

```text
/mnt/workspace/litengjie/data/vendor/PowerPaint/
  powerpaint/models/BrushNet_CA.py
  powerpaint/models/unet_2d_condition.py
  powerpaint/pipelines/pipeline_PowerPaint_Brushnet_CA.py
  powerpaint/utils/utils.py

/mnt/workspace/litengjie/data/models/PowerPaint-v2-1/
  realisticVisionV60B1_v51VAE/
  PowerPaint_Brushnet/diffusion_pytorch_model.safetensors
  PowerPaint_Brushnet/pytorch_model.bin
```

All model components load with `local_files_only=True`. The backend temporarily
adds the repository to `sys.path`, loads the custom UNet, BrushNet, text encoder,
tokenizer wrapper, PowerPaint BrushNet pipeline, and UniPC scheduler, then
restores the import path. It registers `P_ctxt`, `P_shape`, and `P_obj` with ten
vectors per task token. `close()` drops the pipeline and clears the CUDA cache
when CUDA is available.

## Input Manifest

The benchmark consumes an explicit JSONL manifest and never scans a production
run. Every optional value is still represented explicitly:

```json
{
  "sample_id": "clip1-e1-bottom",
  "clip_uid": "clip1",
  "entity_id": "e1",
  "reference_type": "subject",
  "entity_phrase": "a person in a red coat",
  "source_rgba_path": "/mnt/workspace/litengjie/data/inputs/entity.png",
  "context_rgb_path": null,
  "completion_mask_path": null,
  "completion_sides": ["bottom"],
  "completion_start_ratio": 0.45
}
```

`completion_mask_path` must be present and may be `null`. When non-null, it must
be an absolute PNG path under `/mnt/workspace/litengjie/data/`.
`completion_sides` is non-empty, unique, and normalized to the stable order
`top`, `bottom`, `left`, `right`. `completion_start_ratio` is within `[0, 1]`.
The source must be an RGBA PNG with non-empty binary alpha. An optional context
must be a same-size RGB PNG or JPEG. Manifest, source, context, and explicit
mask paths must resolve under `/mnt/workspace/litengjie/data/`.

## Canvas And Completion Mask Modes

The source is placed without resizing on a deterministic canvas. Only requested
sides are extended by `canvas_expand_ratio`; left and top extension determine
the recorded source offset. The optional context fills transparent pixels in the
original source rectangle. Without context, those pixels and all new canvas
areas are white.

When `completion_mask_path` is `null`, the benchmark uses directional mode. For
each requested side, the completion zone starts at
`completion_start_ratio` along the visible alpha bounding box and continues to
that canvas edge. Its perpendicular span includes deterministic lateral
padding. Multiple directional zones are unioned. Visible alpha pixels are then
forced to mask value zero.

When `completion_mask_path` is non-null, the same directional settings still
determine canvas expansion and source offset, but the supplied mask completely
replaces the generated directional mask. `completion_start_ratio` remains in
the manifest and result for compatibility but does not affect explicit-mask
pixels. Different manifest records may use different explicit masks; there is no
global CLI mask option.

An explicit mask must be an L-mode PNG whose dimensions exactly match the final
expanded canvas. It is never resized, thresholded, or antialiased. Its only
allowed values are 0 and 255. It must be non-empty, smaller than the full
canvas, disjoint from visible source pixels, adjacent to their one-pixel
dilation, and different from the complete transparent region.

Every final L-mode mask must be binary, non-empty, smaller than the full canvas,
adjacent to a one-pixel dilation of the visible entity, and different from the
set of all transparent pixels. `255` permits generation and `0` preserves the
baseline. No SAM3 or free-form mask UI is used.

## Model Space And PowerPaint Tasks

Canvas and mask are resized with preserved aspect ratio to a model space whose
short side is approximately 640 and whose dimensions are multiples of eight.
RGB uses Lanczos and mask uses nearest-neighbor interpolation. The output is
restored to the expanded canvas size rather than cropped to the source size.

Before inference, pixels inside the model-space mask are zeroed and the mask is
passed to PowerPaint as RGB. The v2-1 task-token mapping is:

| Strategy | promptA | promptB | negative A | negative B |
| --- | --- | --- | --- | --- |
| `text_guided` | `P_obj` | `P_obj` | `P_obj` | `P_obj` |
| `shape_guided` | `P_shape` | `P_ctxt` | `P_shape` | `P_ctxt` |

The deterministic completion and negative prompts are passed only as `promptU`
and `negative_promptU`. Both `tradoff` and `tradoff_nag` use `fitting_degree`.
The default candidate order is text-guided seeds 0 and 17, followed by
shape-guided seeds 0 and 17.

## Pixel Contract And Hard Gates

Two forced compositing layers constrain every generated candidate:

1. every pixel outside the completion mask is restored from the baseline;
2. every visible source pixel is restored exactly from the original RGBA RGB.

The source file itself remains byte-identical. A candidate must also be RGB,
have the expanded canvas dimensions, change pixels inside the completion mask,
contain non-constant completion content, contain valid pixels, and preserve mask
connectivity and bounds. Any hard-gate failure skips the Qwen judge.
For explicit mode, the published `completion_mask.png` is also checked for
pixel equality with the input, and the input mask SHA-256 must remain unchanged
through the run.

## Structured Review And Selection

The Qwen judge receives four images: the source on white, baseline completion
canvas, binary completion mask, and completed RGB candidate. Its strict schema
checks exact visible-source preservation, continuation of the same entity,
identity, exactly one entity, structural plausibility and usefulness, absence of
reconstructed occluders and new salient entities, boundary quality, and final
reference usability. `accept` is valid if and only if all ten booleans are true.

Unsupported strict `json_schema` requests fall back to `json_object`. Malformed
output receives a bounded structured repair request and otherwise fails closed.
All configured candidates and reviews are retained. The earliest candidate that
passes both hard gates and Qwen review is recorded as accepted; diagnostics are
not used to quality-rank later candidates.

## Execution And Output

The first pilot should contain only a few manually identified local or
incomplete references with explicit completion directions.

```bash
python -m tools.run_v3_reference_completion_benchmark \
  --manifest /mnt/workspace/litengjie/data/reference_completion.jsonl \
  --benchmark-root /mnt/workspace/litengjie/data/reference_completion_benchmarks/run-001 \
  --powerpaint-repo /mnt/workspace/litengjie/data/vendor/PowerPaint \
  --checkpoint-dir /mnt/workspace/litengjie/data/models/PowerPaint-v2-1 \
  --judge-model /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
```

The benchmark root must be a new directory strictly below
`/mnt/workspace/litengjie/data/reference_completion_benchmarks/`. Existing roots
fail closed. Each sample is assembled in a temporary sibling directory and
published with one rename; exceptions remove the temporary directory.
All source, context, and explicit mask files are validated before the benchmark
root is created. One invalid explicit mask therefore aborts the whole manifest
without running PowerPaint or Qwen and without leaving output directories.

```text
<benchmark_root>/<sample_id>/
  source_rgba.png
  source_white.png
  context_rgb.png                 # only when supplied
  baseline_canvas.png
  visible_mask.png
  completion_mask.png
  candidate_text_guided_seed_0.png
  candidate_text_guided_seed_17.png
  candidate_shape_guided_seed_0.png
  candidate_shape_guided_seed_17.png
  review_text_guided_seed_0.json
  review_text_guided_seed_17.json
  review_shape_guided_seed_0.json
  review_shape_guided_seed_17.json
  result.json
```

The expanded RGB candidate is an experiment artifact, not a replacement for the
source RGBA reference. No output is written to `r2v_v3_runs`,
`r2v_v3_datasets`, `selected/`, or the public dataset tree.
`result.json` records `completion_mask_mode` as `directional` or `explicit`.
Explicit mode also records the resolved mask source path and SHA-256;
directional mode records `null` for both fields.

## Final Evaluation

The completed pilot produced a negative result for identity-preserving entity
completion with PowerPaint v2-1.

### Directional Text-Guided Evaluation

The first evaluation used `strategy=text_guided`, seed `0`,
`fitting_degree=0.55`, and the broad directional bottom mask. The candidate was
rejected because it generated multiple new salient people instead of continuing
the original visible entity.

### Explicit Shape-Guided Evaluation

The second evaluation used `strategy=shape_guided`, seeds `0` and `17`,
`fitting_degree=0.80`, and a narrow explicit bottom mask. Both candidates were
rejected. Each generated a separate complete person rather than a connected
continuation of the source entity; identity, clothing, proportions, and
structure were not continuous with the preserved source pixels.

### Frozen Benchmark Evidence

The two existing server-side output directories under
`/mnt/workspace/litengjie/data/reference_completion_benchmarks/` whose
`result.json` files match the two configurations above are frozen benchmark
evidence. They must not be renamed, overwritten, regenerated in place, or used
as production references. Their images, logs, model files, and benchmark
artifacts remain outside this repository and are not committed.

### Decision

PowerPaint v2-1 is not suitable for identity-preserving entity completion in
this pipeline. The narrow explicit mask did not solve the entity-insertion
failure, and the failures cannot be attributed only to one seed or to the broad
directional mask. Prompt, seed, mask, and fitting-degree searches are therefore
closed for this backend, and it will not be integrated into production.

Production references continue to use source-faithful RGBA artifacts. Reference
coverage should preferentially come from real-frame donors and same-parent
cross-pairing. When no qualifying donor exists, no generated completion may be
published.
