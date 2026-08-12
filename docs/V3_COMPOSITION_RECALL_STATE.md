# V3 Composition Recall State

Last updated: 2026-08-12

## Repository State

- Repository: `TjieLee/R2V_DATA_V2`
- Development branch: `feature/v3-composition-recall-v1`
- Parent branch: `feature/v3-boogu-reference-edit`
- Verified branch point and pre-implementation HEAD:
  `60aaa3877f8e928e5d7eb309950a406d070d4344`
- Current implementation HEAD: the commit containing this state document
- Validated production behavior baseline:
  `f4cdec251095ba3fd70c57f0a4082c58e5a67101`
- H3 branch `feature/h3-audio-binding-v1`: untouched
- Existing prod5000 runs: untouched

## Implemented Decisions

1. Composition-balanced entity proposal is additive and opt-in. Default
   annotation requests use the original system prompt exactly.
2. Object phrase rescue is opt-in, deterministic, and limited to one retry per
   object. It never retries a subject or group.
3. Severe subject/object collisions use the existing duplicate comparator.
   Failed rescue rejects the object, never the valid subject.
4. Parent-stratified source selection uses a local RNG and fails if the limit
   cannot be met without violating the per-parent cap.
5. Selection and rescue evidence stays in separate run diagnostics rather than
   expanding `ClipRecord`.
6. The composition audit is read-only and adds no quality gate.

## Production Compatibility

Every new behavior defaults to `default`, `off`, or `sequential`. Coverage
remains 7/10, candidate count remains 3, crop padding remains 0.08, conservative
prefilter and Qwen guards remain unchanged, Boogu is unchanged, and same-parent
cross-pair remains disabled.

## Known Risks and Follow-up

- A complementary annotation scan may modestly increase entity proposal count,
  but it is not a quota and cannot force a hallucinated type.
- A phrase retry can still fail or collapse to a subject; diagnostics expose
  those outcomes and publication remains fail-closed for the object.
- Parent-stratified selection can fail when fewer eligible parent groups exist
  than the requested limit under the configured cap.
- Large internal holes and missing structure in references remain a separate
  integrity problem. They are intentionally not addressed in this experiment.

## Validation Boundary

This pass uses fake annotation and segmentation backends only. The required
local test, Ruff, and diff checks are recorded in the implementation commit and
review response. No Qwen, SAM3, CUDA, Boogu, production data, or 300-clip pilot
was run.

Validated commands for this implementation:

```bash
python -m pytest -q
python -m ruff check .
git diff --check
```

The full local suite completed with 1,500 passing tests and one pre-existing
Pillow deprecation warning.
