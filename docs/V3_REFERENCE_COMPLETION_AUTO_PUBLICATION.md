# V3 localized completion automatic publication benchmark

This benchmark evaluates whether a Qwen `localized_raw` completion can be
published as a generated entity reference without per-sample manual approval.
It is an offline, fail-closed benchmark. It is not part of
`run_pipeline_v3.py`, does not update production artifacts, and must not be
connected to production before a larger mixed-entity pilot is complete.

## Reference priority

The intended future production priority is:

1. an already qualifying source/self reference;
2. a real same-clip or same-parent donor;
3. a Qwen localized generated fallback that passes every publication gate;
4. the original local reference, or no qualifying reference.

A generated fallback never replaces an existing qualifying real reference.
This benchmark records that boundary but does not change pairing, cross-pair,
instruction, export, production schemas, or any existing run or dataset.

## Immutable input contract

The JSONL manifest contains one object per sample:

```json
{
  "sample_id": "sample-001",
  "clip_uid": "video_001_clip_0001",
  "entity_id": "e1",
  "reference_type": "subject",
  "entity_phrase": "the person in a blue coat",
  "source_rgba_path": "/mnt/workspace/litengjie/data/.../source.png",
  "localized_result_path": "/mnt/workspace/litengjie/data/.../result.json"
}
```

The whole batch is preflighted before the output run root is created. The
preflight requires:

- an existing RGBA PNG with non-empty binary alpha and exact-white transparent
  RGB;
- a localized result with `backend=qwen_image_edit_2511` and
  `mode=localized_raw`;
- a passed localized hard-check report with no failed check or reason;
- matching source path and SHA-256 in the manifest, localized result, and file;
- an existing RGB PNG candidate whose SHA-256 matches the localized result;
- every manifest, source, result, and candidate path under
  `/mnt/workspace/litengjie/data`.

Any preflight error aborts the entire batch and creates no benchmark run root.
The source RGBA and localized candidate are copied for diagnostics but are
never modified.

## Output contract

The default run directory is:

```text
/mnt/workspace/litengjie/data/
  reference_completion_publication_benchmarks/<run_id>/
```

Each sample is built in a temporary sibling directory and atomically renamed:

```text
<sample_id>/
  source_rgba.png
  candidate_rgb.png
  candidate_mask.png
  review_identity.json
  review_locality.json
  review_usability.json
  result.json
  candidate_reference.png       # auto_published only
```

`publication_summary.json` records stable counts for processed,
`auto_published`, rejected, reference types, rejection reasons, and each judge.
There is no `manual_review_pending` state.

## Local SAM3 segmentation

The CLI uses only the local SAM3 source and checkpoint:

```text
/mnt/workspace/litengjie/data/vendor/sam3
/mnt/workspace/public/pretrained/facebook/sam3/sam3.pt
```

SAM3 segments the immutable localized RGB candidate with `entity_phrase`. The
adapter is injectable, so tests use a fake segmenter and never load a model.
The adapter disables Hugging Face loading when the installed builder exposes
that option. Missing local paths or an incompatible image API fail explicitly.

Every candidate mask must be an L-mode binary mask at the original candidate
size. Empty and full-canvas masks are rejected. If SAM3 returns multiple masks,
exactly one is selected with this deterministic ordering:

1. confidence descending;
2. foreground area descending;
3. original result index ascending.

The full ranking provenance is stored in `mask_metrics`.

## Deterministic gates

For `subject` and `object`, the default component gates are:

- largest component / all foreground `>= 0.90`;
- second component / all foreground `<= 0.05`.

`group` has separate explicit thresholds and allows multiple natural members.
It rejects a significant component that is spatially isolated from the main
group. The group thresholds are recorded in every result and are not silently
derived from the single-entity settings.

Source alpha and candidate mask metrics include normalized foreground area,
bbox, normalized bbox size, four edge-touch flags, component count, component
areas, and component bboxes. Candidate dimensions may differ from the source,
so publication compares normalized geometry, never absolute pixel area.

The default normalized candidate/source area range is `0.75` through `2.00`.
The explicit group range is `0.60` through `2.50`. Within that conservative
range, at least one real local improvement is required:

- a source edge touch is removed;
- the candidate fills a nearby local bbox extension while substantially
  overlapping the source bbox; or
- normalized entity coverage increases without abnormal growth.

The background gate examines pixels outside the candidate mask. It records
border mean and standard deviation, outside-mask near-white ratio, and
quantized color diversity. The defaults allow a slight natural shadow or
small gray variation while rejecting a large non-white or complex scene.

## Three independent structured judges

Three deterministic Qwen-VL requests use separate strict schemas:

- **identity**: same subject/object/group, no redesign, drift, or extra
  instance;
- **locality**: only a real missing local part was repaired, with no broad
  redraw, composition expansion, new scene, or unrelated content;
- **usability**: the segmented RGBA has clean boundaries and plausible
  geometry, anatomy, material, or group structure, with no ghosting,
  duplicates, fractures, fragments, extra limbs, text, logo, or watermark.

The prompts include reference-type-specific subject, object, and group
semantics. Uncertainty must be represented by `certain=false`, which cannot
coexist with an accepting verdict. A request error, invalid JSON, schema error,
or uncertain result fails closed. There is no majority vote: all three judges
must accept.

## Generated RGBA

Only unanimous success after every deterministic gate creates
`candidate_reference.png`. Its RGB comes directly from the localized candidate
where the selected mask is 255. RGB is exactly `(255, 255, 255)` where the mask
is zero, and alpha is the exact binary mask. The file retains the candidate's
original dimensions. There is no resize, interpolation, upscale, or fixed
1024-pixel canvas. The PNG is reopened to verify RGBA mode, dimensions, binary
alpha, transparent white RGB, and SHA-256 before the sample is published.

Any failed gate produces `status=rejected`, a null published reference, and no
`candidate_reference.png`. An unexpected per-sample processing error removes
the first temporary directory, publishes a deterministic rejection
transaction, and continues with remaining preflighted samples.

## CLI

```bash
python tools/run_v3_reference_completion_publish.py \
  --manifest /mnt/workspace/litengjie/data/.../publication.jsonl \
  --run-id pilot-001 \
  --judge-model /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
```

All thresholds have CLI/configuration equivalents and are serialized in
`result.json`. The CLI may point all three judges at the same local vLLM
endpoint, but still issues three independent structured requests.

## Pilot boundary

Run a mixed pilot of at least 50-100 subject, object, and group samples before
considering production integration. A non-blocking random audit may be useful
for measuring gate precision, but audit completion and human approval are not
publication dependencies. Until that pilot is evaluated, this remains an
offline publication benchmark only.
