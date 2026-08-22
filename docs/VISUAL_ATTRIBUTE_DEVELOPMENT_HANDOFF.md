# Final Visual / Subject Attribute Development Handoff

Updated: 2026-08-23

This is the final handoff for the frozen Visual/reference-image development
line. Subject Attribute development is complete. The next active development
line is Audio/H3 elsewhere; do not continue tuning Visual or Subject Attributes
by default.

## Repository State

```text
repository: TjieLee/R2V_DATA_V2
branch: feature/v3-subject-attributes-v1
code freeze: 51fef9d44bb1372b4afad5fed9795d5c3d46bda7

frozen Visual branch: feature/v3-runtime-integrity-v1
frozen Visual HEAD: 87bd4e06107d7f56df550979b0e96515cb70f911
core Visual freeze baseline: 3cfb11fdd1fbe4a5bbad02a775097d8ab3097288
```

Later docs-only commits may move branch `HEAD`; the algorithm/code freeze remains
`51fef9d...`.

## Hard Isolation

- Annotation production is frozen. Do not change its tools, schema, prompts, or
  entity-selection semantics from this line. See
  `ANNOTATION_ENTITY_PRODUCTION.md`.
- Audio/H3 belongs to a separate branch. Do not merge, rebase, cherry-pick, or
  implement Audio/H3 work here.
- Frozen Visual thresholds, coverage 7/10, pairing semantics, SAM3 behavior,
  Boogu removal semantics, and production shard/cursor rules are not tuning
  targets.
- Use a new correctness cycle only when new real production evidence identifies
  a bug.

## Final Subject Attribute Flow

```text
eligible retained human owner
  -> one Qwen discovery
  -> owner Top3 existing Visual candidate frames
  -> frame-local SAM3
  -> deterministic ownership / geometry
  -> one owner-batched raw Qwen review
  -> complete and usable: raw RGBA
  -> semantic + owner correct and repair recommended:
       Boogu completion
       -> one Qwen SubjectAttributeCompletionReview
       -> accept: native Boogu RGB, selected_completed
       -> reject/failure:
            raw usable: raw RGBA fallback
            raw unusable: reject
```

The completion prompt remains:

```text
把图片中破损、缺失或不完整的区域补充完整。
```

There is no `segmentation_prompt`, generated-image SAM3 resegmentation, mask
union, alpha restoration, generated foreground extraction, or second repaired
`SubjectAttributeReview`. GME is disabled and not part of production.

Duplicate discovery types are normalized first-wins before the unchanged
strict uniqueness schema. Discovery still performs exactly one Qwen call.

## Runtime Layout

```text
GPU 0-3: Qwen3-VL-32B-Instruct, BF16, TP1 x DP4
GPU 4:   Boogu background removal
GPU 5+7: shared two-process SAM3 pool for temporal SAM and attribute probes
GPU 6:   Boogu reference_edit and Subject Attribute completion
GME:     disabled

Qwen endpoint: http://127.0.0.1:8000/v1
Qwen max model length: 49152
runtime.qwen_max_inflight: 4
```

Keep `NO_PROXY`/`no_proxy`, the SAM3 `PYTHONPATH`, worker-local GPU isolation,
and no global parent `CUDA_VISIBLE_DEVICES` as documented in
`SERVER_ENVIRONMENT_RUNBOOK.md`.

## Final Evidence

### Difficult four-clip canary

The `canary-e2e4-jea-20260821-140636-s000000000-000000003` run included
`42038d7dc619cfa7bebee437` / `e1` / `a3` / `upper_clothing`. Raw review
requested repair; Boogu completion passed; the native 1120 x 928 RGB result was
selected and published byte-for-byte with `synthetic=true`. All completion-SAM
and repaired-final-review counters remained zero.

The archaeological lesson is not "derive a shorter SAM phrase". The Boogu
output was good and completion SAM damaged it, so completion SAM was removed.

### Fixed random 200

The `e2e200-random-seed20260821-20260821-143232` run at `8d0c8ad...` exported
114 Visual samples and 220 Visual references. Subject Attributes accepted 161
attribute references across 101 enriched samples. Completion selected 106 of
140 attempted completion candidates; that ratio is not overall dataset yield.
The run exposed duplicate discovery types and one atomic-state failure.

### Targeted fresh 10

After `51fef9d...`, the 9 duplicate-type clips plus the atomic-write clip passed
with 10/10 `clip.json` files, zero orphan/temp leftovers, zero runtime failures,
and zero duplicate-discovery failures.

Detailed metrics and validation caveats are authoritative in
`V3_SUBJECT_ATTRIBUTES_STATE.md`.

## Code Lineage

### `71cddcd0b5e99d249ce18427e31bb5a3b9046274`

`fix(v3): use boogu attribute completion output directly`

- removed completion SAM resegmentation;
- made the completed crop the direct Boogu RGB output.

### `31477f2303fde1246ef6386af660ef4efe828bcb`

`fix(v3): simplify attribute completion acceptance`

- removed the repaired final `SubjectAttributeReview`;
- made Qwen completion acceptance directly select `selected_completed`;
- persisted `completion_review`;
- made the canary force GPU4 Boogu removal.

### `8d0c8ad1ac8d7910221c5a52c1d1756868b0b924`

`fix(v3): compact completed attribute references`

- established raw RGBA / completed RGB production modes;
- published completed attributes with `synthetic=true`;
- preserved attribute bytes through compaction.

### `51fef9d44bb1372b4afad5fed9795d5c3d46bda7`

`fix(v3): normalize discovery and isolate atomic temps`

- normalized duplicate `attribute_type` values first-wins;
- gave every atomic JSON writer a unique temporary filename;
- cleared both real issues in the targeted regression.

## Integration-Facing Contracts

Audio/H3 integration should join the stable production interface rather than
reconstructing Visual internals.

```text
stable machine identity: clip_uid / sample_id
canonical sample schema: r2v.v3.production_sample.1
reference kinds: subject / object / group / background / attribute

attribute provenance:
  attribute_id
  owner_entity_id
  attribute_type
  source_frame_index
  synthetic

raw attribute: RGBA, synthetic=false
completed attribute: RGB, synthetic=true

target video: processed public-dataset MP4
target video copying: never
```

## Validation Record

`51fef9d...` added five passing tests and no new failing set relative to its
Windows baseline. Its focused candidate result was 130 passed with the same two
Windows pipe/select failures as baseline; full candidate was 1803 passed / 96
failed versus 1798 passed / 96 failed at baseline. No full Linux pytest is
claimed for `51fef9d...`. The last known all-green Linux full run was
`8d0c8ad...`: 1894 passed, 1 warning.

## Final Decision

Visual/Subject Attribute/reference-image development is frozen at
`51fef9d44bb1372b4afad5fed9795d5c3d46bda7`. Continue only for a newly
demonstrated correctness bug; otherwise integrate Audio/H3 against the stable
contracts above.
