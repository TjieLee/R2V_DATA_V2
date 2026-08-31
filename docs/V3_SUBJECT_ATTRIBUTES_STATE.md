# V3 Subject Attributes State

Last updated: 2026-08-28

This is the authoritative production contract for Visual V3 subject attributes.
It supersedes older design notes, benchmark interpretations, and chat history.

## Freeze identity

```text
repository: TjieLee/R2V_DATA_V2
Visual/reference branch: feature/v3-subject-attributes-v1
final Visual/reference code freeze: d056c32b76db4b3d7c0358b38e996e7a91a288d1

frozen original Visual branch: feature/v3-runtime-integrity-v1
frozen original Visual HEAD: 87bd4e06107d7f56df550979b0e96515cb70f911
core original Visual algorithm baseline: 3cfb11fdd1fbe4a5bbad02a775097d8ab3097288
```

Later docs-only commits may advance the branch HEAD. They do not advance the
algorithm freeze above. Annotation production is frozen. Audio/H3 is developed
on a separate branch and is not developed on this Visual branch.

## Subject Attribute production flow

The supported types are:

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

Each eligible human owner has at most three discovered attributes. Enrichment
reuses that owner's existing Top3 Visual candidates, considers at most two
different source candidates, and uses only single-frame SAM3 probes. It never
runs temporal attribute tracking. GME is disabled.

The raw owner-batched review has six hard acceptance flags:

```text
matches_attribute
owner_binding_correct
recognizable
characteristic_appearance_visible
usable_as_attribute_condition
sufficient_source_evidence
```

`structure_complete` and `completion_recommended` are diagnostics only. They
cannot invalidate a review and do not control the Boogu route.

Completion is eligible only for:

```text
headwear
accessory
upper_clothing
lower_clothing
dress_or_skirt
```

Completion remains disabled for `face`, `hair`, `glasses`, `shoes`, and `bag`.
`face` is also blocked by code policy even when a legacy server-local config
still lists it in `eligible_types`; it cannot create a completion seed or call
the Boogu backend or comparative completion review.

Face has a separate fixed publication route after all existing SAM3 geometry,
ownership, semantic, and six-flag raw review gates pass:

```text
accepted raw face -> materialize and review source RGB bbox
  -> bbox accept: publish bbox, final_selection=bbox
  -> bbox reject, judge failure/unavailable, or materialization failure:
       publish the already accepted raw alpha, final_selection=raw
  -> neither accepted raw alpha nor accepted bbox: reject
```

Bbox preference never lets an unaccepted raw face bypass upstream quality or
evidence gates. Face bbox is non-synthetic; raw face remains binary-alpha RGBA.

The exact prompt is:

```text
补全成完整的{entity_name}，去除掉破碎的部分，不添加其他内容
```

The fixed mapping is:

```text
headwear -> 帽子
accessory -> 配饰
upper_clothing -> 衣服
lower_clothing -> 下装
dress_or_skirt -> 裙子
```

The completion comparison is:

```text
Image 1: raw alpha attribute
Image 2: source RGB bbox, identity evidence only
Image 3: Boogu candidate
```

For eligible types, the same physical item or component is a hard gate. A
modest real improvement is enough for `candidate_better_than_alpha`; an
equivalent candidate falls back to alpha. Boogu output is reviewed directly and
then receives one frame-local Attribute completion SAM3 probe using the original
grounding prompt. All usable masks are unioned, only very small floating
components are removed, and the Boogu RGB plus cleaned binary mask is
materialized as RGBA. The cleaned artifact must pass deterministic quality
checks before the existing comparative completion review can accept it. There
is no temporal SAM3 tracking or second repaired attribute review. Legacy
completed RGB sidecars remain readable by the compactor.

For non-face types, if `sufficient_source_evidence=false`, Boogu must not
hallucinate missing evidence. The pipeline tries candidate 2. If both sources
remain insufficient, bbox is the last resort. Bbox is otherwise the last resort
only when there is no accepted completion and no accepted alpha. It has
anti-copy-paste gates and rejects tiny or weak semantic evidence. If bbox
rejects, the attribute rejects.

Fresh Attribute generated backgrounds are disabled. Their shared variants slot
is always:

```json
{
  "image_path": null,
  "status": "unavailable",
  "reviewed": false,
  "review_status": "not_applicable",
  "reason": "attribute_background_disabled_by_policy",
  "synthetic": true
}
```

## Publication contract

Raw attributes remain binary-alpha RGBA PNGs. Accepted completion images remain
native Boogu RGB PNGs. Bbox references are RGB PNGs. The production compactor
validates the mode selected by `final_selection`, materializes the original
bytes without conversion, and records completed references as synthetic.

`final_selection` may be `raw`, `completed`, or `bbox`. Attribute provenance in
`r2v.v3.production_sample.1` includes `attribute_id`, `owner_entity_id`,
`attribute_type`, `source_frame_index`, and `synthetic`.

## Frozen deterministic gates

```text
MIN_ATTRIBUTE_AREA_PIXELS = 16
MIN_ATTRIBUTE_LONG_SIDE_PIXELS = 4
hair minimum long side = 192
headwear minimum long side = 128
maximum attribute / owner area ratio = 0.85
wrong-owner overlap rejection = 0.50
clothing owner-like area ratio = 0.78
clothing strip aspect ratio = 3.0
duplicate mask IoU = 0.90
```

Fragmentation rejects when any condition is true:

```text
largest_component_ratio < 0.70
OR second_largest_component_ratio > 0.20
OR significant_component_count > 3
```

## Runtime allocation

```text
GPU 0-3: Qwen3-VL-32B-Instruct, BF16, TP1 x DP4
GPU 4: Boogu background removal
GPU 5 + 7: shared SAM3 pool for main segmentation and attribute probes
GPU 6: Boogu reference_edit and Attribute completion
GME: disabled

Qwen endpoint: http://127.0.0.1:8000/v1
Qwen max model length: 49152
runtime.qwen_max_inflight: 4
```

## Final real-model evidence

The latest fixed-100 run was:

```text
run: e2e100-verify-324a29a-20260825-234558
real-model commit: 324a29aebcf4b573cab59332d337ec9d10ad9deb
```

Reference edit:

```text
background attempted/accepted: 0/0
completion attempts: 5
candidate 1 accepted: 3
candidate 2 attempts: 0
fallback alpha: 2
entities accepted: 95
entities rejected: 0
```

Reference integrity:

```text
entities accepted: 84
entities rejected: 17
bbox fallback attempted/accepted: 1/1
```

Attributes:

```text
accepted attributes: 84
completion attempts: 70
completion accepted: 51
completion Qwen review rejects: 19
second candidate attempts: 21
candidate 1 accepted: 46
candidate 2 accepted: 5
bbox fallback attempted/accepted: 3/0
background variants: 0/0
```

The validator fix after `622be6807490d57aebbda76b6dcf102beded7aff`
was motivated by a one-owner/three-attribute batch-loss bug exposed by this
canary. It passed the full local unit suite; the fixed-100 real-model run was not
rerun for that validator-only correction.

At the earlier `d7f3d6b99e5da02bd8ef275ab53cd47cd649cfa0` freeze, the
candidate-2 source-frame provenance correction was validated locally with 368
focused tests and 1,991 full tests passing (one warning), plus
`git diff --check` and compileall. No additional GPU or real-model canary was run
for that provenance-only correction.

The explicit face bbox-first policy unfreeze at
`d056c32b76db4b3d7c0358b38e996e7a91a288d1` was validated locally with 148
focused Subject Attribute tests and 2,000 full tests passing (one warning), plus
compileall and `git diff --check`. Ruff reported the same 69 pre-existing
findings on the clean parent and the updated tree, with no new findings from
this change. No Qwen, SAM3, Boogu, GPU, or real-model canary was run.

## Freeze decision

Do not change prompts, gates, routing, model topology, or schemas on this branch
without new production evidence and an explicit unfreeze decision. See
`V3_VISUAL_AUDIO_INTEGRATION.md` before integrating Audio/H3.
