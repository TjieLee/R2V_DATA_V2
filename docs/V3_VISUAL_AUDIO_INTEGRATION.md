# Visual V3 and Audio/H3 Integration Contract

Last updated: 2026-08-31

This is the compatibility handoff Audio/H3 developers must read before
consuming current Visual/reference outputs.

## Verified branch relationship

The refs inspected for this handoff were:

```text
repository: TjieLee/R2V_DATA_V2

Visual/reference branch: feature/v3-subject-attributes-v1
Visual/reference code HEAD inspected by the original handoff: d7f3d6b99e5da02bd8ef275ab53cd47cd649cfa0

Standalone Visual Stage2 branch: feature/v3-sam3-production-v1
validated Entity Mask orchestration HEAD: 8645db869892ea98a6599467882efc48b4eb7414

Audio branch: feature/h3-audio-jea-qwen3-v1
Audio remote HEAD verified on 2026-08-31: 4e831e3f4c29f52df3909af719abf94300b4f633

Visual/Audio merge base for the inspected Audio branch:
71276f976ed178242abe4db3e66e3fecce357832
```

The Visual/reference branch may gain docs-only commits after the code HEAD
above. The original frozen Visual branch is
`feature/v3-runtime-integrity-v1` at
`87bd4e06107d7f56df550979b0e96515cb70f911`; its core algorithm baseline is
`3cfb11fdd1fbe4a5bbad02a775097d8ab3097288`.

The standalone Stage2 branch is a separate production-orchestration line around
frozen Visual components. It must not be treated as a new cross-modal public
schema.

At the refs used by the original Visual/Audio integration handoff, both
branches had the same `r2v_data_v2/v3/production_export.py` Git blob:

```text
3de3ce9277160f147903ccca804b6f28b1ec63c1
```

Therefore there was no `r2v.v3.production_sample.1` schema delta between those
inspected Audio and final Visual code refs.

## 2026-08-31 standalone Stage2 orchestration isolation

The recent Entity Mask production work was intentionally isolated from Audio/H3.
The three hardening commits after the first formal Entity Mask launcher are:

```text
1ccd2b833fc9f7c89031062f2883e7983005f600
511e834e590b1517e0e61670c2a95e244573ac88
8645db869892ea98a6599467882efc48b4eb7414
```

From the Entity Mask launcher baseline
`12737b791c636f441191841d91eaae431bd0d34e` through
`8645db869892ea98a6599467882efc48b4eb7414`, the changed production code is
limited to standalone Visual Stage2 orchestration and its tests/docs:

```text
r2v_data_v2/v3/pre_qwen_production.py
tools/run_v3_entity_mask_auto.py
tools/run_v3_entity_mask_worker.py
tests/test_v3_entity_mask_static.py
tests/test_v3_pre_qwen_production.py
docs/ENTITY_MASK_PRODUCTION.md
```

No `r2v_data_v2/h3/*` file, Audio runner, Audio schema, or
`production_export.py` was changed by this sequence. The Audio branch is a
separate diverged branch and does not receive these commits unless they are
explicitly merged/cherry-picked.

The recent Stage2 changes therefore have no direct runtime effect on the
current Audio/H3 branch. Their scope is:

```text
multi-node/multi-GPU Entity Mask scheduling
static whole-shard ownership
100-row durable Stage2 resume orchestration
SAM3 worker persistence/session reuse orchestration
shared-filesystem startup identity
worker lifecycle/signal cleanup
production docs/tests
```

If an Audio integration branch later cherry-picks or merges
`pre_qwen_production.py`, re-run the Audio/H3 test suite because that becomes an
explicit code integration. Do not merge the entire Visual Stage2 branch merely
to inherit its cluster scheduler; the stable Visual/Audio data contract does
not require that scheduler.

## Stable integration API

Audio/H3 should consume compacted production records with schema:

```text
r2v.v3.production_sample.1
```

Do not treat Subject Attribute owner artifacts, review payloads, enrichment
metrics, standalone Entity Mask checkpoints, 100-row execution chunks, or
internal sidecars as the final cross-branch API.

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

Standalone Stage2 parallelism is deliberately absent from this schema. Node
count, GPU count, `RANK`, `WORLD_SIZE`, global worker ID, shard ownership, and
100-row chunk identity are internal execution details and must never become
Audio training-sample identity.

## Internal sidecar delta since the merge base

These changes are relevant only if Audio reads Visual internal sidecars rather
than compacted production output.

1. `ReferenceEditEntityState` adds
   `accepted_base_image_path: Optional[str]`. When
   `default_variant="accepted_base"`, `default_image_path` points there rather
   than to `variants.alpha`, `variants.bbox`, or
   `variants.generated_background`.

2. `ReferenceVariantsManifestRecord` remains
   `r2v.v3.reference_variants.1`, but Subject/Object may now use
   `default_variant="accepted_base"` with a non-null
   `accepted_base_image_path`. An older Audio parser that requires null for
   entity records will fail.

3. `SubjectAttributeReview` adds required
   `sufficient_source_evidence`. Its six hard accepted flags are the five
   existing quality/binding flags plus that evidence flag.
   `structure_complete` and `completion_recommended` are diagnostics only.

4. `SubjectAttributeCompletionReview` no longer uses the old generic
   `BooguCompletionReview` shape. It contains:

   ```text
   same_physical_attribute
   original_visible_details_preserved
   no_wrong_new_instance
   no_duplicate_component
   no_unrelated_content
   no_structural_distortion
   target_clear_and_prominent
   candidate_better_than_alpha
   certain
   reason
   verdict
   ```

5. `SubjectAttributeBboxReview` adds
   `no_strong_owner_pose_or_scene_leakage`.

6. `SubjectAttributeRecord.final_selection` now allows `raw`, `completed`, and
   `bbox`.

7. `OwnerEnrichmentMetrics` adds:

   ```text
   attribute_source_candidates_considered
   attribute_second_candidate_attempts
   attribute_completion_candidate1_accepted
   attribute_completion_candidate2_accepted
   attribute_bbox_fallback_attempts
   attribute_bbox_fallback_accepted
   attribute_background_variants_attempted
   attribute_background_variants_accepted
   ```

8. Generic `BooguCompletionReview` renamed the verdict-driving completion flag
   from `missing_parts_plausibly_completed` to
   `candidate_better_than_source`. The latest Visual reader normalizes the
   legacy name.

9. Reference-edit completion metadata adds retry provenance:

   ```text
   completion_attempt_index
   completion_source_candidate_id
   completion_source_frame_index
   comparison_source_image_path
   comparison_source_image_hash
   ```

   These fields are not public `ProductionSample` fields.

10. Several internal version strings did not bump despite these compatible or
    normalized changes:

    ```text
    r2v.v3.subject_attributes.1
    r2v.v3.subject_attribute_owner.1
    r2v.v3.enriched_sample.1
    r2v.v3.reference_variants.1
    ```

Do not infer old/new internal payload shape from `schema_version` alone. The
latest Visual reader can normalize supported legacy sidecars, but an older
Audio reader is not necessarily forward-compatible with latest Visual output.

## Current `single_run_export` risk

`r2v_data_v2/h3/visual_production_source.py` on the inspected Audio branch
directly imports:

```python
from r2v_data_v2.v3.subject_attributes import EnrichedSample
```

Its `single_run_export` path strictly parses:

```text
subject_attributes/enriched_samples.jsonl
```

Reading a latest Visual run through an older Audio checkout may therefore raise
`ValidationError` when internal fields are added or replaced even though the
stable production schema is unchanged.

Production integration should use `compacted_production` and
`r2v.v3.production_sample.1`. If `single_run_export` must remain supported, the
Audio integration branch must synchronize the latest Visual Subject Attribute
read schemas and compatibility normalizers, then add a regression fixture from
a latest Visual `enriched_samples.jsonl` payload.

## Branch ownership

Annotation production remains frozen. Do not develop Audio/H3 on the Visual
branch, and do not change Visual prompts, thresholds, model topology, Stage2
scheduler, or public production schema merely to make an old internal-sidecar
parser accept a newer payload.

For standalone Entity Mask server operation, use
`ENTITY_MASK_PRODUCTION.md`. For Audio/H3 integration, consume the stable
compacted production interface rather than Entity Mask internal artifacts.
