# Visual V3 and Audio/H3 Integration Contract

Last updated: 2026-08-31

This is the compatibility and ownership contract for the integrated Visual V3
and Audio/H3 production line. Read it before changing cross-modal interfaces or
consuming Visual production outputs from H3.

## Current integration state

The authoritative refs for the completed integration are:

```text
repository: TjieLee/R2V_DATA_V2

integration branch:
feature/visual-audio-integration-v1

integration HEAD:
56cf57b933ff4614643aedab57e67824504037d5

initial integration merge commit:
4dc497403f0aa65114a543a81a851a310a55c497

latest Visual refresh merge commit:
56cf57b933ff4614643aedab57e67824504037d5

Visual source branch:
feature/v3-sam3-production-v1
Visual source HEAD:
21a6fe5eebc33f8d35b8434600d860c2550ba8e6

Audio source branch:
feature/h3-audio-caption-multibackend-v1
Audio source HEAD:
efa04668715081348df81ee25746272599a0df12

Visual/Audio merge base:
71276f976ed178242abe4db3e66e3fecce357832
```

The initial integration merge has the Audio source commit and the then-current
Visual source commit `8001a2f3801e50c42f64e9135569b7a65017b73b` as its two
parents and completed without textual merge conflicts. H3 then received one
narrow compatibility commit that accepts the Visual face `bbox` attribute
selection in its private projection and adds regression coverage. After Visual
advanced by five additional commits, the integration branch merged Visual again
at `21a6fe5eebc33f8d35b8434600d860c2550ba8e6`; that refresh also completed
without conflicts and changed no H3 production code.

The original frozen Visual branch remains
`feature/v3-runtime-integrity-v1` at
`87bd4e06107d7f56df550979b0e96515cb70f911`; its core original algorithm
baseline is `3cfb11fdd1fbe4a5bbad02a775097d8ab3097288`. The current Subject
Attribute face policy is represented by the later Visual algorithm commit
`d056c32b76db4b3d7c0358b38e996e7a91a288d1` and subsequent Visual
postprocessing/publication fixes through `21a6fe5eebc33f8d35b8434600d860c2550ba8e6`.
All are preserved in the integration branch.

LASER ASD experimental files are not part of this integration line. The
production ASD direction remains the existing LR-ASD/TalkSet Audio path.

## Stable public integration API

Both source branches and the integration branch use the same
`r2v_data_v2/v3/production_export.py` blob:

```text
3de3ce9277160f147903ccca804b6f28b1ec63c1
```

The public cross-modal schema remains:

```text
r2v.v3.production_sample.1
```

Audio/H3 should prefer compacted production records using this schema. Do not
bump the public schema merely because Visual internal sidecars or production
orchestration evolve.

Stable `ProductionSample` fields:

```text
sample_id
clip_uid
target_video
t2v_caption
r2v_instruction
references
source
```

Stable `ProductionReference` fields:

```text
image_id
image_index
kind
entity_id
attribute_id
owner_entity_id
attribute_type
image_path
source_frame_index
scope
visible_region
synthetic
```

Reference `kind` values are:

```text
subject
object
group
background
attribute
```

Production reference indexes are contiguous. Instruction labels may appear in
any numeric order, but every expected index must appear and no unknown index is
allowed.

Do not treat Subject Attribute owner artifacts, review payloads, enrichment
metrics, standalone Entity Mask checkpoints, 100-row execution chunks, worker
identities, or other internal sidecars as the public Visual/Audio API.

## Visual Stage1/Stage2 coexistence and runtime isolation

The integration branch intentionally contains the complete latest Visual
production line, including standalone Stage1 Annotation production and
standalone Stage2 Entity Mask production.

The Stage2 production code includes:

```text
r2v_data_v2/v3/pre_qwen_production.py
tools/run_v3_pre_qwen_auto.py
tools/run_v3_pre_qwen_batch.py
tools/run_v3_entity_mask_auto.py
tools/run_v3_entity_mask_worker.py
```

and the associated tests/docs.

The integrated Visual production line preserves:

```text
static whole-shard multi-node/multi-GPU scheduling
100-row durable checkpoint/resume
persistent SAM3 workers
clip_reset_v1 session reuse
vectorized binary-mask RLE
launcher lifecycle hardening
shared-filesystem startup identity checks
```

These modules coexist with H3 but are not part of H3 runtime. Audio/H3
production code must not import or invoke the standalone Visual Stage2
scheduler merely because both now live on the same branch.

Node count, GPU count, `RANK`, `WORLD_SIZE`, global worker ID, shard ownership,
100-row chunk identity, and SAM3 worker/session state remain Visual execution
details and must never become Audio training-sample identity.

The shared V3 primitives that H3 legitimately consumes, such as
`production_export.py`, selected V3 schemas, and `mask_codec.py`, remain normal
library dependencies. In particular, H3 uses the latest Visual vectorized RLE
decoder rather than maintaining a separate mask codec.

## Latest Visual Subject Attribute semantics

The integration branch preserves the latest Visual Subject Attribute semantics.
Important internal deltas since the original Visual/Audio merge base include:

1. `ReferenceEditEntityState` adds
   `accepted_base_image_path: Optional[str]`. When
   `default_variant="accepted_base"`, `default_image_path` points there rather
   than to `variants.alpha`, `variants.bbox`, or
   `variants.generated_background`.

2. `ReferenceVariantsManifestRecord` remains
   `r2v.v3.reference_variants.1`, but Subject/Object may use
   `default_variant="accepted_base"` with non-null accepted-base provenance.

3. `SubjectAttributeReview` includes required
   `sufficient_source_evidence`. `structure_complete` and
   `completion_recommended` remain diagnostics rather than separate public
   production fields.

4. `SubjectAttributeCompletionReview` uses the current attribute-specific
   completion review shape rather than the older generic completion payload.

5. `SubjectAttributeBboxReview` includes
   `no_strong_owner_pose_or_scene_leakage`.

6. `SubjectAttributeRecord.final_selection` allows exactly:

   ```text
   raw
   completed
   bbox
   ```

7. Internal Subject Attribute metrics and completion metadata gained additional
   source-candidate, retry, bbox, resume, and publication-routing provenance
   without requiring a public `ProductionSample` schema bump.

Several Visual internal version strings did not bump despite compatible or
normalized payload changes. Do not infer exact old/new internal payload shape
from an internal `schema_version` alone.

## Face bbox-first policy

Face attributes are special and the current Visual policy must not be reverted
by Audio-side compatibility work.

`face` is excluded from
`SubjectAttributeCompletionConfig.eligible_types`, so face does not use Boogu
completion.

The accepted face flow is:

```text
raw semantic / ownership / review gates pass
    -> materialize source RGB bbox candidate
    -> bbox review
        -> accept: final_selection = "bbox"
        -> reject/failure/unavailable: final_selection = "raw"
```

The bbox path does not bypass raw semantic, ownership, or review gates.

A face selected with `final_selection="bbox"` remains a real source crop and is
not synthetic.

## H3 Visual reader ownership

H3 owns a projection of the Visual fields it actually needs. It must not couple
itself to every Visual-internal diagnostic field.

`r2v_data_v2/h3/visual_clip_contract.py` provides an H3-owned projection for
`clip.json` fields and ignores unrelated Visual-internal sections.

`r2v_data_v2/h3/visual_production_source.py` similarly uses private projection
models for `single_run_export` enrichment input. These projection models use
`extra="ignore"` so that unrelated Visual diagnostics do not break H3.

Do not replace these H3-owned projections with strict parsing of the complete
Visual internal models unless the public integration contract explicitly
changes.

## `compacted_production` and `single_run_export`

`compacted_production` is the preferred production integration path:

```text
compacted production samples.jsonl
    -> r2v.v3.production_sample.1
    -> ProductionSample validation
    -> H3 VisualProductionInventory
```

`single_run_export` remains a compatibility/debug path for directly consuming a
single Visual run and its Subject Attribute enrichment sidecar.

The previous forward-compatibility risk has been resolved on the integration
branch. H3 no longer imports and strictly validates the complete Visual
`EnrichedSample` model. It uses its own projection and accepts exactly:

```text
final_selection = raw | completed | bbox
```

The integration regression includes a latest-Visual-style accepted face
attribute with:

```text
attribute_type = "face"
final_selection = "bbox"
default_variant = "bbox"
```

and verifies that H3 preserves the owner, frame index, and image path, ignores
unconsumed Visual-internal diagnostics, and publishes:

```text
synthetic = false
```

Legacy `raw` and `completed` selections remain supported.

## Branch ownership after integration

The integration branch is the common line for cross-modal compatibility and
combined production-reader validation.

Ownership remains modular:

```text
r2v_data_v2/v3/*
    -> Visual semantics/runtime ownership

r2v_data_v2/h3/*
    -> Audio/H3 semantics/runtime ownership

r2v_data_v2/v3/production_export.py
    -> stable cross-modal public contract
```

New Audio algorithms should still be developed on dedicated Audio feature
branches and integrated deliberately. New Visual algorithms should still be
developed on dedicated Visual feature branches and integrated deliberately.
Do not use the integration branch as a reason to couple their internal
orchestration.

Do not change Visual prompts, thresholds, model topology, Stage2 scheduling, or
public production schema merely to make an old Audio-side internal parser work.
Fix the H3 projection when a new Visual internal field is legitimately part of
the H3 compatibility surface.

For standalone Entity Mask server operation, use
`docs/ENTITY_MASK_PRODUCTION.md`. For Audio/H3 production integration, prefer
the stable compacted production interface described above.
