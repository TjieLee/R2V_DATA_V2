# V3 Reference Background-Inpainting Benchmark

## Status And Scope

This is an offline experiment. It is not a V3 production pipeline stage and is
not included in `run_pipeline_v3.py`. The production entity reference remains
the source-faithful RGBA artifact. Benchmark results must be reviewed before any
production integration is proposed.

The experiment generates only the transparent background around an existing
entity reference. Reference viewpoint augmentation, entity redrawing, entity
completion, pairing changes, instruction changes, and dataset export changes
are outside this benchmark.

## Pixel Contract

The source must be an RGBA PNG with binary alpha values (`0` or `255`) and a
non-empty foreground. The backend receives the original dimensions. It may pad
an input to a model alignment multiple, but it must crop the generated image
back to the original coordinates. Resizing, upscaling, and fixed 1024 canvases
are prohibited.

Only pixels where source alpha is zero are taken from model output. Before a
candidate is published, code copies the original source RGB back into every
pixel where alpha is 255 and verifies exact equality. The original RGBA file is
never overwritten.

## Input Manifest

The benchmark reads an explicit JSONL manifest and never scans a production
run. Every line has this form:

```json
{
  "sample_id": "clip1-e1",
  "clip_uid": "clip1",
  "entity_id": "e1",
  "reference_type": "subject",
  "entity_phrase": "a person in a red coat",
  "source_rgba_path": "/mnt/workspace/litengjie/data/inputs/entity.png"
}
```

The manifest and every source path must resolve under
`/mnt/workspace/litengjie/data/`. The output must be an explicit run directory
strictly below
`/mnt/workspace/litengjie/data/reference_inpaint_benchmarks/`. Paths escaping
these roots and any output under a `selected/` directory are rejected.

## Execution

The optional Qwen Image Edit backend loads the base model from a local path with
`local_files_only=True`. It does not load the Object-Remover LoRA and is not a
fallback for the production removal backend. A separate OpenAI-compatible Qwen
VL endpoint provides structured candidate review.

```bash
python -m tools.run_v3_reference_inpaint_benchmark \
  --manifest /mnt/workspace/litengjie/data/reference_benchmark.jsonl \
  --benchmark-root /mnt/workspace/litengjie/data/reference_inpaint_benchmarks/run-001 \
  --model-path /mnt/workspace/litengjie/data/models/Qwen-Image-Edit-2511 \
  --judge-model /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
```

The default seeds are `0` and `17`. All configured candidates are retained for
offline comparison and evaluated in seed order. A candidate must pass every
local hard check and receive an accepting structured review. The earliest
accepted seed wins. There is no fallback that accepts the source or a failed
candidate.

## Hard Checks And Review

Every publishable candidate must be a same-size RGB PNG with finite `uint8`
pixels. Source foreground RGB must be pixel-identical, and generated background
pixels must not be all white, all black, or another single constant value. RGB
publication removes any possibility of residual transparency at the entity
boundary. Candidate SHA-256 values are recorded.

The Qwen judge receives only:

1. the original RGBA entity displayed on white;
2. the generated RGB candidate;
3. the binary alpha mask;
4. the entity phrase and reference type.

It returns strict JSON. Unsupported `json_schema` requests fall back to
`json_object`; malformed structured output receives a bounded repair attempt and
otherwise fails closed.

## Output

Each sample is assembled in a temporary sibling directory and published with
one directory rename. Exceptions remove the temporary directory, so a partial
sample is never exposed.

```text
<benchmark_root>/<sample_id>/
  source_rgba.png
  baseline_white.png
  candidate_seed_0.png
  candidate_seed_17.png
  review_seed_0.json
  review_seed_17.json
  result.json
```

`result.json` records source and candidate hashes, hard-check results, judge
verdicts, the earliest accepted selection, runtime, and rejection reasons. No
benchmark artifact is written into `r2v_v3_runs`, `r2v_v3_datasets`, `selected/`,
an existing clip directory, or the public dataset tree.
