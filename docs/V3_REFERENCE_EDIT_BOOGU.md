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

SAM3 may re-segment a generated candidate for presence, instance-count, growth,
and fragmentation diagnostics. Those masks are review-only and cannot modify
the candidate.

## Artifacts

```text
clips/<clip_uid>/reference_edit/<entity_id>/
  source_rgba.png
  source_input_rgb.png
  completion_candidate_1k.png
  completion_metadata.json
  background_candidate_1k.png
  background_metadata.json
  final_reference_1k.png
  final_metadata.json
  rejection.json
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

The worker is the only repository file that imports torch or Boogu, and those
imports occur once during worker startup. macOS development and unit tests use
fake backends and do not load Boogu, CUDA, weights, or real data.

## Production routing

The candidate judge records `image_quality` and one completeness route:
`complete`, `repairable`, `local_usable`, `severely_incomplete`, or
`fragmented`. `repairable` enters entity completion, `complete` may receive a
background, `local_usable` keeps the source-faithful local reference, and the
two severe outcomes are rejected. Qwen and SAM3 review generated candidates;
SAM3 masks are review-only. An accepted candidate becomes the entity's
`final_reference_1k.png`. Rejections follow the configured `keep_source` or
`reject_entity` policy.

## Validation

Run local CPU checks with POSIX shell commands:

```bash
python -m pytest tests/test_v3_reference_edit_boogu.py -q
python -m pytest -q
python -m ruff check .
git diff --check
```
