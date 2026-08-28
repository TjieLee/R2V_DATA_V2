# V3 Boogu Reference Completion Contract

Last updated: 2026-08-28

## Freeze identity

```text
repository: TjieLee/R2V_DATA_V2
Visual/reference branch: feature/v3-subject-attributes-v1
final Visual/reference code freeze: d056c32b76db4b3d7c0358b38e996e7a91a288d1

frozen original Visual branch: feature/v3-runtime-integrity-v1
frozen original Visual HEAD: 87bd4e06107d7f56df550979b0e96515cb70f911
core original Visual algorithm baseline: 3cfb11fdd1fbe4a5bbad02a775097d8ab3097288
```

Later docs-only commits may advance branch HEAD. This document describes the
frozen production behavior at the code freeze above.

## Fresh Subject/Object behavior

Boogu is used only for repairable `complete_entity` references. Complete and
locally usable references remain canonical alpha. Fresh entity-background
generation is disabled and its generated-background variant remains unavailable
with reason `entity_background_disabled_by_policy`.

The exact generation prompt is:

```text
Complete the missing or broken parts of the same target entity: "{entity_phrase}".
Preserve its identity, appearance, colors, materials, proportions, and style.
Do not add another entity or unrelated content.
Remove broken fragments and keep uncertain completion simple and consistent with visible evidence.
```

The first attempt uses the canonical source candidate. When it is rejected or
not better and an alternate source exists, the second attempt uses candidate 2.
The alternate attempt records:

```text
completion_attempt_index
completion_source_candidate_id
completion_source_frame_index
comparison_source_image_path
comparison_source_image_hash
```

An accepted candidate-2 reference publishes its actual alternate
`source_frame_index`. Legacy metadata without that field falls back to the
canonical reference frame for read compatibility.

The final routing is:

```text
candidate 1 completion + comparative Qwen
  -> accepted and better: completion 1
  -> otherwise candidate 2 when available
       -> completion 2 + three-image comparative Qwen
       -> accepted and better: completion 2
       -> otherwise canonical alpha
```

After reference integrity, a failed selected completion falls back to canonical
alpha integrity. Bbox is attempted only after alpha integrity fails. Completion
does not fall directly to bbox.

## Review and SAM gates

The comparative Qwen review requires the same physical entity, identity, and
semantics. A modest real improvement is enough; equivalent output returns to
canonical alpha. Translation, recentering, moderate scale, crop, and layout
changes are allowed. Warped structure, duplicate or wrong instances, tiny
targets, and extreme-corner placement reject.

SAM review still hard-rejects target-not-found, multiple target instances,
severe fragmentation, and backend failure. For `complete_entity`, source-relative
area growth, scale, and center shift are diagnostic only because valid repair
may complete missing structure. Legacy background-operation compatibility may
retain its old gates, but fresh production never routes to that operation.

SAM masks used by Subject/Object reference edit are review evidence. They do not
alter the generated image artifact. The persistent Boogu worker loads once on
GPU 6 and serves reference edit serially per worker while the pipeline remains
concurrent across stages.

## Subject Attribute completion reuse

Attribute completion reuses the persistent GPU 6 Boogu worker but has a separate
exact prompt:

```text
补全成完整的{entity_name}，去除掉破碎的部分，不添加其他内容
```

Mapping:

```text
headwear -> 帽子
accessory -> 配饰
upper_clothing -> 衣服
lower_clothing -> 下装
dress_or_skirt -> 裙子
```

Face, hair, glasses, shoes, and bag do not enter completion. Face is hard-blocked
in code even if a legacy config still names it as eligible: it creates no seed
and calls neither Boogu nor comparative completion Qwen. An accepted raw face
instead prefers a reviewed source RGB bbox and falls back to its accepted raw
alpha on bbox reject, failure, or unavailability. Bbox cannot rescue a face that
failed the existing raw hard gates.

Eligible non-face Attribute comparison uses raw alpha, source RGB bbox as
identity-only evidence, and the generated candidate. It must preserve the same
physical item or component. A modest improvement is enough and an equivalent
candidate returns to alpha.

Accepted Attribute Boogu output is published directly as RGB PNG. It receives
no completion SAM3, alpha restoration, foreground extraction, or bbox pass.
Fresh Attribute background generation is disabled with reason
`attribute_background_disabled_by_policy`.

## Artifacts and provenance

Attempt artifacts remain durable and auditable, including the canonical and
alternate source inputs, generated candidates, Qwen/SAM review JSON, and
per-attempt metadata. Candidate-retry layouts may include names such as:

```text
alternate_source_2.png
completion_candidate_2_1k.png
metadata_2.json
```

Internal retry provenance is not part of the public
`r2v.v3.production_sample.1` schema. The stable public reference carries its
selected `source_frame_index` and normal entity/reference provenance.

## Frozen runtime

```text
Boogu code: /mnt/workspace/litengjie/data/vendor/Boogu-Image
Boogu python: /mnt/workspace/litengjie/data/venvs/boogu-image/bin/python
Boogu model: /mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708
GPU: 6
instruction rewrite: disabled for completion
thinking: disabled
```

Do not change this contract without new production evidence and an explicit
unfreeze decision.
