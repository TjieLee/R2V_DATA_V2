# V3 Composition Recall V1

## Scope

This opt-in experiment improves the chance that a clip containing both a useful
subject and a useful object proposes and retains both types. It does not loosen
coverage, candidate ranking, crop geometry, reference quality, background
guards, or reference-edit publication policy.

The implementation branch is `feature/v3-composition-recall-v1`. Its parent is
`feature/v3-boogu-reference-edit`, and its verified branch point is
`60aaa3877f8e928e5d7eb309950a406d070d4344`. The validated production behavior
baseline remains `f4cdec251095ba3fd70c57f0a4082c58e5a67101`.

## Measured Baseline

The prod5000 output contains 1,189 final samples:

| Composition | Count | Share |
| --- | ---: | ---: |
| Subject only | 678 | 57.02% |
| Object only | 443 | 37.26% |
| Subject and object | 37 | 3.11% |

Samples containing an object account for 40.62%, and samples containing a
subject account for 60.47%. Final entity counts are 783 subjects, 534 objects,
and 32 groups. Subject and object entity-level retention was nearly equal
(about 34.7% and 34.3% from annotation to final), so this experiment does not
change coverage or pair thresholds.

## Opt-in Behavior

`qwen.annotation.entity_selection_mode: composition_balanced_v1` appends one
short complementary scan to the existing system prompt. It is explicitly a
preference, not a subject/object quota. The annotation schema and sanitizer are
unchanged.

`sam3.object_rescue_mode: phrase_retry_v1` permits exactly one object-only
tracking retry with `entity.phrase` after an initial `not_found`, or before the
existing severe subject/object duplicate gate discards a collapsed object
track. The existing duplicate thresholds remain the sole collision definition.
Subjects and groups are never retried by this policy. An unresolved cross-type
collision rejects only the object.

Rescue diagnostics are written under the new run root at
`diagnostics/object_rescue/` and reconciled into
`diagnostics/object_rescue.jsonl`. They do not enter `clip.json`.

`source.selection_mode: parent_stratified_random_v1` uses a local seeded RNG,
shuffles parent groups and clips deterministically, and enforces a per-parent
cap. The selected indices and identities are recorded atomically in
`source_selection.json` under the new run root. The public source dataset is
read-only.

## Frozen Contracts

- `coverage.required_visible_frames: 7`
- `pair.max_candidates_per_entity: 3`
- `pair.crop_padding_ratio: 0.08`
- `pair.reference_prefilter_mode: conservative_v1`
- `pair.background_final_guard_mode: qwen_v1`
- `reference_edit.scale_collapse_fallback_guard_mode: qwen_v1`
- `pair.same_parent_fallback_enabled: false`
- Existing Boogu prompts and publication policy

The known problem of references with unnatural internal holes or missing
structure is a separate integrity project. No topology filter or additional
Qwen quality gate is included here.

## Deterministic 300-clip Pilot

Use a fresh run and export root. The relevant configuration is:

```yaml
source:
  start_index: 0
  limit: 300
  allow_full_run: false
  selection_mode: parent_stratified_random_v1
  random_seed: 20260812
  max_clips_per_parent: 1

qwen:
  annotation:
    entity_selection_mode: composition_balanced_v1

sam3:
  object_rescue_mode: phrase_retry_v1

coverage:
  required_visible_frames: 7

pair:
  max_candidates_per_entity: 3
  crop_padding_ratio: 0.08
  reference_prefilter_mode: conservative_v1
  background_final_guard_mode: qwen_v1
  same_parent_fallback_enabled: false

reference_edit:
  scale_collapse_fallback_guard_mode: qwen_v1
```

Run the pilot only after review, with the server-specific paths and model
services supplied in the complete config:

```bash
python run_pipeline_v3.py \
  --config /mnt/workspace/litengjie/data/configs/v3-composition-recall-300.yaml \
  --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,instruct,export \
  --profile
```

No real pilot, Qwen call, SAM3 run, or Boogu run is part of this code pass.

## Review Audit

After a pilot finishes, generate the read-only composition report outside its
run root:

```bash
python tools/audit_v3_entity_composition.py \
  --run-root /mnt/workspace/litengjie/data/r2v_v3_runs/<run-id> \
  --output-root /mnt/workspace/litengjie/data/r2v_v3_audits/<run-id>-composition \
  --contact-sheets
```

The tool writes `summary.json`, mixed subject/object CSV and JSON lists, and
optional contact sheets without changing source artifacts.
