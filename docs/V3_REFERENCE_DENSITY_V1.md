# V3 Reference Density V1

## Scope

This opt-in experiment increases proposal recall for stable, independently
controllable visual references without weakening downstream quality gates. It
starts from `9b2e3b1913612be5ff1010b18424f7896a98300a` on
`feature/v3-composition-recall-v1` and is implemented separately on
`feature/v3-reference-density-v1`.

The previous `composition120` run produced 119 annotation-ready clips with 123
annotation entities: 77 subjects and 46 objects. It published 39 samples with
47 final entity references and 52 total references including backgrounds. Its
measured final entity-reference density was therefore 1.21 per accepted sample.

## Opt-in Behavior

`qwen.annotation.entity_selection_mode: reference_dense_v1` replaces only the
entity-selection section of the annotation prompt with one coherent dense
selection policy. It asks Qwen to inspect for additional stable entities after
finding the clearest primary entity. Stable, visually distinct worn or attached
objects may be proposed, but no quota is imposed and weak evidence must produce
fewer proposals. Tiny arbitrary details and a small set of generic object
phrases are discarded.

The shared annotation schema can represent eight entities. Application-level
publication limits remain mode-specific:

- `default`: at most 5 entities
- `composition_balanced_v1`: at most 5 entities
- `reference_dense_v1`: at most 8 entities

The dense prompt and deterministic renderer support `{{entity_1}}` through
`{{entity_8}}`. No relation, apparel, salience, control-role, or other ontology
field was added.

`pair.entity_geometry_mode: type_aware_v1` changes only the initial tiny-content
gate:

| Reference type | Minimum bbox area | Minimum long side |
| --- | ---: | ---: |
| subject | 128 x 128 | 128 |
| object | 64 x 64 | 96 |
| group | 128 x 128 | 128 |

`ContentGeometry.area_pixels` remains bounding-box area, not foreground-mask
pixel count. The default `legacy` mode retains the previous 128 x 128 area and
128-pixel long-side requirements for every reference type.

The parent-stratified random manifest scan now parses and groups every source
record without checking every video on the shared filesystem. It preserves the
same local RNG, sorted parent list, parent shuffle, and within-parent shuffle.
Only candidates needed to satisfy the limit are checked. A missing candidate
falls through to the next shuffled clip from that parent, then later parents.
For an all-valid source manifest, selection is exactly compatible with the
previous implementation for the same selection parameters.

## Frozen Quality Contracts

- `coverage.required_visible_frames: 7`
- `pair.max_candidates_per_entity: 3`
- `pair.crop_padding_ratio: 0.08`
- `pair.reference_prefilter_mode: conservative_v1`
- `pair.background_final_guard_mode: qwen_v1`
- `reference_edit.scale_collapse_fallback_guard_mode: qwen_v1`
- `pair.same_parent_fallback_enabled: false`
- `sam3.object_rescue_mode: phrase_retry_v1`
- Existing candidate fragmentation and duplicate thresholds
- Existing Qwen reference judge and Boogu publication behavior

No subject rescue or new integrity judge is included. References with large
holes or structurally poor crops remain a known follow-up issue.

## Read-only Audit

The composition audit now reports annotation and final-reference density,
type counts, synthetic versus real references, full versus local references,
and background reference count. Optional review sheets are grouped under
`subjects`, `objects`, `all_entities`, and `multi_entity_samples`.

```bash
python tools/audit_v3_entity_composition.py \
  --run-root /mnt/workspace/litengjie/data/r2v_v3_runs/<run-id> \
  --output-root /mnt/workspace/litengjie/data/r2v_v3_audits/<run-id>-density \
  --contact-sheets
```

The output root must remain outside the source run. The audit never updates
`clip.json`, selected references, or exports.

## Exact 120-clip Pilot Configuration

Apply this configuration to a fresh run and export root. Do not change the
listed quality fields for the comparison pilot.

```yaml
source:
  start_index: 0
  limit: 120
  allow_full_run: false
  selection_mode: parent_stratified_random_v1
  random_seed: 20260812
  max_clips_per_parent: 1

qwen:
  annotation:
    entity_selection_mode: reference_dense_v1

sam3:
  device: cuda
  object_rescue_mode: phrase_retry_v1

coverage:
  required_visible_frames: 7

pair:
  entity_geometry_mode: type_aware_v1
  max_candidates_per_entity: 3
  crop_padding_ratio: 0.08
  reference_prefilter_mode: conservative_v1
  background_final_guard_mode: qwen_v1
  same_parent_fallback_enabled: false

reference_edit:
  scale_collapse_fallback_guard_mode: qwen_v1
```

Use the reviewed complete config at the server path below. Physical GPU 5 is
selected at process launch; SAM3 must continue to receive `device: cuda`.

```bash
CUDA_VISIBLE_DEVICES=5 python run_pipeline_v3.py \
  --config /mnt/workspace/litengjie/data/configs/v3-reference-density-120.yaml \
  --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,instruct,export \
  --profile
```

This code pass does not launch the command or run Qwen, SAM3, Boogu, CUDA, or
production data.
