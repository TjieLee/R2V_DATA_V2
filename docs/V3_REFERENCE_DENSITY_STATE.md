# V3 Reference Density State

Last updated: 2026-08-12

## Repository State

- Repository: `TjieLee/R2V_DATA_V2`
- Parent commit: `9b2e3b1913612be5ff1010b18424f7896a98300a`
- Parent experiment branch: `feature/v3-composition-recall-v1`
- Development branch: `feature/v3-reference-density-v1`
- H3 branch `feature/h3-audio-binding-v1`: untouched
- Existing completed runs and exports: untouched

## Implementation State

1. `reference_dense_v1` uses a mode-specific STEP 1 and supports at most eight
   annotation entities. Existing annotation modes remain capped at five.
2. The dense sanitizer removes only a small conservative set of generic object
   phrases. It does not add an object-name ontology.
3. `type_aware_v1` preserves subject and group geometry while allowing a lower
   object-only bbox threshold. `legacy` remains the default.
4. Parent-stratified random selection avoids full-dataset video existence
   checks while preserving all-valid seeded selection compatibility.
5. Source-selection provenance includes scan, parent, filesystem-check, and
   missing-candidate diagnostics.
6. The read-only audit reports annotation and final-reference density and emits
   four review-sheet groups.

## Five-entity Assumption Audit

The repository was searched for `MAX_ANNOTATION_ENTITIES`, `entity_5`, `five`,
`range(5)`, `range(1, 6)`, `<= 5`, and `< 6`.

- The schema capacity changed from five to eight.
- Annotation prompt, count words, sanitizer limit, and placeholder ceiling are
  mode-aware.
- `RunStorage.write_annotation` enforces five for the two existing modes and
  eight only for dense mode.
- Pairing tokens, instruction bindings, rendering, and export validation were
  already derived from entity lists and required no numeric expansion.
- Unrelated five-element kernels and test fixtures were left unchanged.

## Compatibility and Risks

Default configuration behavior is unchanged: annotation remains `default`,
geometry remains `legacy`, and source selection remains `sequential` unless
explicitly configured. `composition_balanced_v1` keeps its exact additive prompt
and five-entity application limit.

The experiment increases proposal recall, not publication permissiveness. It
does not alter temporal coverage, candidate count, crop padding, prefilter,
fragmentation, duplicate, background, cross-pair, reference-edit, or export
gates. Structurally poor references that pass the existing judge remain a known
separate integrity issue.

## Validation Boundary

Validation in this branch is CPU-only and uses fake/local test artifacts. No
Qwen request, SAM3 model, Boogu model, CUDA process, real pilot, completed run,
or export tree is accessed or modified. The focused annotation/config/pair/
composition suite passed 323 tests. The complete suite passed 1,518 tests with
one pre-existing Pillow deprecation warning; Ruff and `git diff --check` also
passed.
