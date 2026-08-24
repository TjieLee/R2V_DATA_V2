# V3 Boogu Reference Edit

This is the production V3 post-pair `reference_edit` stage. It treats Boogu
output as a newly generated reference image, not as a repair layer for the
canonical reference. Legacy Qwen completion source and artifacts remain
readable, but production pairing does not invoke that fallback when this stage
is enabled.

## Fixed server layout

```text
Boogu code:   /mnt/workspace/litengjie/data/vendor/Boogu-Image
Boogu Python: /mnt/workspace/litengjie/data/venvs/boogu-image/bin/python
Boogu model:  /mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708
Revision:     hotfix-1k-20260708
```

The parent process invokes the configured Python executable directly. The
worker does not run `conda activate`, `source activate`, or `micromamba
activate`. The model is loaded from the local path with network access disabled.

## Image contract

`selected/<entity_id>.png` remains immutable. A source with alpha is composited
onto pure white to create the sole RGB model input. No alpha mask or SAM mask is
sent to Boogu.

The output dimensions are derived from the source aspect ratio at approximately
one megapixel and aligned to multiples of 16. Width and height are passed
explicitly to the worker. Boogu must return one RGB PNG at exactly those
dimensions. The result is never resized or cropped back to the source size.

For `complete_entity`, instruction rewriting is enabled. For
`add_entity_background`, instruction rewriting is disabled. Both operations
produce a complete new image. The accepted `final_reference_1k.png` is a
byte-for-byte copy of the native worker candidate. There is no mask paste-back,
foreground restoration, protected-core restoration, or pixel-identity gate.

The production instructions are fixed:

```text
图片中只有一个实体，实体是“{entity_phrase}”。补全残缺的部分。不要引入新的实体，风格保持一致。如果补全不了，则只保留最能表示该实体的部分，去除零散且不合理的部分。
给图像添加符合风格的背景，不要增加任何实例。
```

SAM3 may re-segment a generated candidate for presence, instance-count, growth,
and fragmentation diagnostics. Those masks are review-only and cannot modify
the candidate.

Before generation, source content geometry is measured from the nonzero alpha
bbox, never from the full transparent or white canvas. Sources with bbox area
below `16384` pixels or longest bbox side below `128` pixels fall back to the
unchanged source with `tiny_source_entity`; no Boogu request is sent. The long
side rule intentionally permits sufficiently large thin objects.

When SAM3 returns a candidate target mask, both bboxes are normalized to their
own image dimensions. Publication requires:

```text
candidate_scale_ratio = candidate_normalized_bbox_area / source_normalized_bbox_area
candidate_scale_ratio >= 0.60
normalized_center_distance <= 0.20
```

Scale collapse and layout shift reject with `entity_scale_collapsed` and
`entity_shifted_off_layout`. If SAM3 cannot produce a mask, no geometry values
are fabricated; the existing fail-closed or background `not_found` policy
applies, and Qwen must still affirm subject scale and layout preservation.

Every SAM review records `diagnostics.failure_kind` as `none`, `not_found`,
`multiple_instances`, `excessive_area_growth`, `fragmented`, or
`backend_failure`. Qwen rejection always rejects. A passing SAM review follows
the normal acceptance path. The only SAM failure exception is
`add_entity_background` with an accepted Qwen review and `not_found`; that
candidate may publish with `sam_warning: target_not_found`. Completion
`not_found` and every other SAM failure remain hard rejections.

## Subject Attribute Completion Reuse

Subject Attribute completion reuses the same persistent Boogu infrastructure
and GPU6 capacity, but it is not the normal entity `reference_edit` validation
path.

Its frozen generic prompt is:

```text
把图片中破损、缺失或不完整的区域补充完整。
```

Normal `reference_edit` may use generated-candidate SAM3 masks for review-only
diagnostics as documented above. Subject Attribute completion does not run
generated-image SAM3 resegmentation and does not apply the normal
`reference_edit` SAM diagnostics or geometry gates to the completed image.
Acceptance is determined by `SubjectAttributeCompletionReview`; an accepted
candidate publishes the native Boogu RGB output.

Do not apply normal entity-reference SAM diagnostics, completion masks, alpha
restoration, or geometry gates to Subject Attribute completion unless a future
explicit design changes this contract.

## Artifacts

```text
clips/<clip_uid>/reference_edit/<entity_id>/
  source_rgba.png
  source_input_rgb.png
  completion_candidate_1k.png
  completion_metadata.json
  completion_rejection.json
  background_candidate_1k.png
  background_metadata.json
  background_rejection.json
  final_reference_1k.png
  final_metadata.json
```

Rejected candidates remain diagnostic artifacts. Rejection never changes the
canonical reference and never publishes a composited rescue image. The caller
may keep the canonical reference, try another source candidate, or reject the
entity.

## Persistent worker boundary

The stage starts one worker process before its first eligible entity:

```bash
/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python \
  tools/run_v3_boogu_reference_edit_worker.py \
  --serve \
  --code-root /mnt/workspace/litengjie/data/vendor/Boogu-Image \
  --model-path /mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708 \
  --model-name Boogu-Image-0.1-Edit-Turbo \
  --model-revision hotfix-1k-20260708 \
  --device cuda:0 \
  --seed 0
```

The worker loads the pipeline once, emits a JSONL readiness record, then reads
one request per stdin line and writes one response per stdout line. Every edit
and shutdown message carries a `request_id`. All eligible entities in the stage
reuse this process. The parent sends shutdown after the stage, writes worker
stderr to a separate run log, and fails closed on timeout, process exit,
request-ID mismatch, or invalid JSON. When there are no eligible entities, no
worker is started. `CUDA_VISIBLE_DEVICES` comes from `reference_edit` config.
The native call uses `align_res=false`, and an exposed instruction rewriter is
moved explicitly to the same configured device. A valid per-request JSONL error
rejects only that generation; it does not unload the worker or prevent later
entities from using the already loaded pipeline.

The worker is the only repository file that imports torch or Boogu, and those
imports occur once during worker startup. macOS development and unit tests use
fake backends and do not load Boogu, CUDA, weights, or real data.

## Production routing

The candidate judge records `image_quality` and one completeness route:
`complete`, `repairable`, `local_usable`, `severely_incomplete`, or
`fragmented`. `complete` runs only background generation. `repairable` first
runs completion review, then passes the accepted completion candidate into
background generation using the same worker. `local_usable` runs background
generation but retains its local scope and visibility semantics. The two severe
outcomes do not enter this stage.

Visible truncation through a person's torso, hips, buttocks, legs, or arms, or
through a main object structure, is `repairable` even when the visible mask is
one connected component. A stable local identity view without such truncation
may remain `local_usable`. For legacy local references with no completeness
value, a non-whole recognizable alpha bbox touching the canvas boundary routes
to repairable; an explicit `local_usable` value always wins.

Qwen and SAM3 review each generated candidate; SAM3 masks are review-only. If a
repairable completion passes but background generation fails, the accepted
completion candidate is published byte-for-byte and `final_metadata.json`
records `background_fallback=completion_candidate`. Other rejections follow the
configured `keep_source` or `reject_entity` policy. Background generation is an
optional enhancement, not a mandatory replacement. Its existing structured
Qwen review compares source and candidate and must affirm subject scale, subject
layout, coherent and beneficial background, and that the candidate is
preferable to the source. Final selection is deterministic:

```text
repairable: accepted background > accepted completion > source
complete/local_usable: accepted preferable background > source
```

`final_metadata.json` records `final_selection` and
`final_selection_reason`. A source fallback remains non-synthetic and preserves
the original reference fields and bytes.

## Manual smoke regression checklist

Run these only on the server pilot; the identifiers are review notes and are
not production conditions.

- Extreme tiny-person source: expect `tiny_source_entity` and no Boogu request.
- Both examples where the subject shrank into the upper-left: expect the
  scale/layout gate to reject publication.
- `b35058...` / `e3`: expect `repairable`; completion failure keeps the source.
- `b35058...` / `e1`: an implausible background must not replace the source.
- Duplicate-crab example: duplicate candidate remains a hard rejection.

## Validation

Run local CPU checks with POSIX shell commands:

```bash
python -m pytest tests/test_v3_reference_edit_boogu.py -q
python -m pytest -q
python -m ruff check .
git diff --check
```
