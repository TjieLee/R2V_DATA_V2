# Visual / Subject Attribute Development Handoff

Updated: 2026-08-21

This is the short operational state needed to continue Visual V3 Subject Attribute development without reconstructing prior decisions from chat history.

## 1. Repository state

Repository:

```text
TjieLee/R2V_DATA_V2
```

Active Visual + Subject Attribute branch:

```text
feature/v3-subject-attributes-v1
```

Latest Subject Attribute code baseline:

```text
8211911d522e0e60a6760d76c49466579cbf0a4b
fix(v3): derive attribute completion sam prompt with qwen
```

The documentation commits after `8211911d...` contain no intended algorithm change. To verify that a checked-out HEAD contains the current attribute code baseline:

```bash
git merge-base --is-ancestor \
  8211911d522e0e60a6760d76c49466579cbf0a4b HEAD
```

Frozen Visual reference branch:

```text
feature/v3-runtime-integrity-v1
frozen HEAD: 87bd4e06107d7f56df550979b0e96515cb70f911
core freeze baseline: 3cfb11fdd1fbe4a5bbad02a775097d8ab3097288
```

Audio/H3 is a separate frozen line and must remain isolated:

```text
feature/h3-audio-binding-v1
3c4d28018e41461bd296f6e40d985e35a04c6ab5
```

Never checkout/merge/rebase/cherry-pick Audio/H3 work into the Visual development line.

## 2. Annotation production is frozen

Standalone entity annotation is already in production. Its implementation is not part of current Subject Attribute iteration.

Frozen production tools:

```text
tools/run_v3_annotation_batch.py
tools/run_v3_annotation_auto.py
```

Key commits:

```text
8fdce3c513e95baab1e64917ee851c70c5e407c5
8d196c1415b2e72153209361684066c9f8077485
```

Do not modify these files, annotation schema, or entity-selection semantics during Subject Attribute work. See `docs/ANNOTATION_ENTITY_PRODUCTION.md` for the production runbook.

## 3. Common server commands

Server repository:

```text
/mnt/workspace/litengjie/data/R2V_DATA_V2
```

Update safely:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2

git status --short
git fetch origin feature/v3-subject-attributes-v1
git switch feature/v3-subject-attributes-v1
git pull --ff-only origin feature/v3-subject-attributes-v1

git rev-parse HEAD
git status --short
```

If `git status --short` is non-empty before the pull, stop and inspect local changes first.

Main Python environment:

```text
/mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python
```

Focused attribute tests:

```bash
.venv/bin/python -m pytest tests/test_v3_subject_attributes.py -q
```

Full regression:

```bash
.venv/bin/python -m pytest -q
git diff --check
```

Current `8211911d...` validation reported by the implementation pass:

```text
Subject Attribute focused: 92 passed
Subject Attribute + runtime + shared Boogu: 192 passed
Full pytest: 1901 passed, 1 warning
git diff --check: pass
```

No real GPU/Qwen/SAM3/Boogu canary has been run after `8211911d...` yet.

## 4. Storage boundaries

Normal V3 writable root is intentionally restricted to:

```text
/mnt/workspace/litengjie/data
```

Normal `run_root` and `export_root` must remain separate trees below that root.

Typical trees:

```text
/mnt/workspace/litengjie/data/r2v_v3_configs
/mnt/workspace/litengjie/data/r2v_v3_runs
/mnt/workspace/litengjie/data/r2v_v3_exports
```

Main source dataset roots are under:

```text
/mnt/workspace/public/dataset
```

Treat the public dataset as read-only for normal Visual V3. The standalone annotation production output under `.../entity_annotations` is a narrow explicit exception; do not generalize it to Visual V3.

Do not write or modify `/mnt/workspace/liutao` for R2V development. A colleague's external launch script may live in their own environment, but R2V code, configs, runs, logs, and models stay in the R2V/litengjie roots.

Do not copy the source MP4 corpus into run directories.

## 5. Production dataset paths

Growing source index:

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl
```

Processed clip root:

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped
```

Source-video root:

```text
/mnt/workspace/public/dataset/jea-video/moive-183t-0808
```

The source grows over time. Do not encode a fixed total-record count into the Visual algorithm contract.

## 6. Model and runtime paths

### Qwen

Main Qwen model:

```text
/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
```

Normal local development endpoint:

```text
http://127.0.0.1:8000/v1
```

Current full Visual runtime target uses Qwen max inflight `4`. Do not increase it casually without profiling.

### SAM3

Code:

```text
/mnt/workspace/litengjie/data/vendor/sam3
```

Checkpoint:

```text
/mnt/workspace/public/pretrained/facebook/sam3/sam3.pt
```

For direct ad-hoc Python probes outside the persistent worker, expose the vendor package explicitly:

```bash
PYTHONPATH=/mnt/workspace/litengjie/data/vendor/sam3:${PYTHONPATH:-} \
CUDA_VISIBLE_DEVICES=7 \
.venv/bin/python ...
```

Production/representative Visual layout uses a shared two-worker SAM3 pool on physical GPUs 5 and 7. Attribute frame probes and completion re-segmentation share this pool. Keep eager SAM3 (`sam3_compile_enabled: false`) for this production layout unless a new benchmark justifies a change.

### Boogu

Code:

```text
/mnt/workspace/litengjie/data/vendor/Boogu-Image
```

Python:

```text
/mnt/workspace/litengjie/data/venvs/boogu-image/bin/python
```

Model:

```text
/mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708
```

GPU 6 is the intended persistent Boogu reference-edit / Subject Attribute completion worker in the current Visual layout. Do not launch a second persistent Boogu copy for attribute completion.

Representative full Visual allocation:

```text
GPU 0-3: Qwen service
GPU 4: Boogu background removal
GPU 5: shared SAM3 worker A
GPU 6: Boogu reference edit + attribute completion
GPU 7: shared SAM3 worker B
```

This Visual development allocation is separate from the colleague's multi-node annotation production, where all eight GPUs on those nodes may be used as independent Qwen endpoints.

## 7. Frozen Visual semantics that should not drift casually

Important frozen behavior includes:

```text
sampled frames: 10
coverage requirement: 7 / 10
pair max candidates/entity: 3
crop padding: 0.08
Qwen reference-integrity max_tokens: 1024
```

Do not loosen Visual reference thresholds just to increase yield.

The known good formal 1000-clip canary remains the historical quality baseline:

```text
run:
/mnt/workspace/litengjie/data/r2v_v3_runs/e2e1000-s0-samfix-20260814-101818

config:
/mnt/workspace/litengjie/data/r2v_v3_configs/e2e1000-s0-samfix-20260814-101818.yaml

export:
/mnt/workspace/litengjie/data/r2v_v3_exports/e2e1000-s0-samfix-20260814-101818

result:
1000 raw -> 362 exported samples -> 530 references
36.2% yield, about 1.46 refs/exported sample
```

## 8. Subject Attribute baseline

Supported attribute types:

```text
face
hair
headwear
glasses
upper_clothing
lower_clothing
dress_or_skirt
shoes
bag
accessory
```

Core rules:

- every attribute has an `owner_entity_id`
- only retained ready human subjects are eligible owners
- at most 3 discovered attributes per owner
- attribute SAM3 probes are single-frame only: no tracking, no propagation, no 7/10 rule
- prefer an attribute source frame different from the owner's final subject reference when possible
- raw attribute output is transparent RGBA from RGB + SAM mask; do not invent a background
- fail closed on uncertain ownership/semantics
- GME prefilter is abandoned for production and remains disabled by default

Current deterministic geometry constants include:

```text
MIN_ATTRIBUTE_AREA_PIXELS = 16
MIN_ATTRIBUTE_LONG_SIDE_PIXELS = 4
hair minimum long side = 192
headwear minimum long side = 128
max attribute/owner area ratio = 0.85
wrong-owner overlap reject = 0.50
clothing owner-like area ratio = 0.78
clothing strip aspect ratio = 3.0
duplicate mask IoU = 0.90
```

Fragmentation reject baseline:

```text
largest_component_ratio < 0.70
OR second_largest_component_ratio > 0.20
OR significant_component_count > 3
```

Do not replace this with a blind "keep largest component" rule.

## 9. Attribute completion semantics after `8211911d...`

Boogu prompt is intentionally category-agnostic and frozen:

```text
把图片中破损、缺失或不完整的区域补充完整。
```

Raw review independently judges usability and structural completeness.

Routing remains:

```text
raw usable + complete
  -> use raw, no completion

raw usable + incomplete/recommended
  -> attempt completion
  -> any completion failure falls back to raw

raw unusable + semantic/owner correct + repair recommended
  -> attempt completion
  -> repair must fully pass, otherwise reject

wrong semantic target / wrong owner / hard deterministic geometry failure
  -> no completion, reject
```

The completion path is now:

```text
Boogu generated candidate
  -> Qwen attribute-only completion identity/quality review
     + one concise segmentation_prompt
  -> Qwen reject: fail closed, SAM3 is not called
  -> Qwen accept: exactly one SAM3 completion re-segmentation call
  -> union all non-empty returned SAM masks with logical OR
  -> deterministic size / crop-quality / component / area-growth / bbox-growth checks
  -> repaired subset owner-batched SubjectAttributeReview
```

There is no SAM prompt fallback/retry loop. One completion attempt means at most one completion SAM call.

`SubjectAttributeCompletionReview` is attribute-specific. The shared normal Visual `BooguCompletionReview` schema/prompt/acceptance semantics are not changed by this design.

Normal SubjectAttributeReview call bound remains at most two calls per owner:

```text
1 raw owner batch
+ optional 1 repaired-subset owner batch
```

## 10. Why Qwen now supplies the SAM prompt

A real generated clothing candidate exposed a SAM text-grounding false negative.

Diagnostic run:

```text
/mnt/workspace/litengjie/data/r2v_v3_runs/attribute-canary20-multimask-20260821-184841
```

Case:

```text
clip_uid: 42038d7dc619cfa7bebee437
owner: e1
attribute: a3
candidate: candidate_2
attribute_type: upper_clothing
phrase: gray traditional robe with dark sash
grounding_prompt: gray traditional robe with dark sash worn by the man
```

Boogu generated image was visually useful as a clothing reference, but direct SAM probe results were:

```text
gray traditional robe with dark sash -> 0 masks
gray traditional robe                -> 1 mask, area 600308
traditional robe                     -> 0 masks
robe                                 -> 0 masks
clothing                             -> 1 mask, area 603226
garment                              -> 1 mask, area 604012
```

This demonstrated that the image and SAM3 were both usable; the over-specific completion grounding phrase caused a false zero-mask. The fix is therefore not to loosen SAM gates or add repeated SAM retries. The existing Qwen completion review now emits one short grounding phrase before the single SAM call.

## 11. Current validation state and next action

`8211911d...` has unit/regression validation only. No real GPU/Qwen/SAM3/Boogu canary has been run after that commit.

Next development action:

1. Pull a HEAD containing `8211911d...`.
2. Run a narrow real completion canary first, preferably exercising the known clothing case or another completion-triggered owner.
3. Verify the runtime order `Boogu -> Qwen review -> SAM once` and inspect the emitted `segmentation_prompt`.
4. Verify `completion_selected_completed > 0` on a valid repaired case before expanding the canary.
5. Inspect the resulting repaired attribute image manually for identity, owner correctness, garment/component completeness, contamination, and mask quality.
6. Only after this passes should the canary be expanded. Do not change thresholds merely to force acceptance.

Useful completion metrics:

```text
completion_attempts
completion_selected_completed
completion_fallback_to_raw
completion_postcheck_rejects
completion_sam_zero_mask_rejects
completion_sam_single_mask
completion_sam_multi_mask
completion_sam_masks_returned_total
completion_identity_review_rejects
completion_final_review_rejects
```

The old `attribute-canary20-multimask-20260821-184841` run is diagnostic evidence from the pre-`8211911d` behavior. Do not silently mutate normal RunStorage git-commit identity protections just to reuse an old run under new code.

## 12. Hard isolation checklist

Before every Visual/Attribute commit:

```text
[ ] Annotation production files unchanged unless explicitly doing a versioned production fix
[ ] Audio/H3 untouched
[ ] frozen Visual thresholds not casually changed
[ ] no writes under /mnt/workspace/liutao
[ ] no public-dataset write-policy relaxation
[ ] no MP4 copying
[ ] focused tests pass
[ ] full pytest passes
[ ] git diff --check passes
[ ] real GPU canary status reported explicitly (run or not run)
```
