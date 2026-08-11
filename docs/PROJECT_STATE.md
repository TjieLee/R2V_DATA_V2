# R2V_DATA_V2 Project State

Last updated: 2026-08-11

## Repository

- Repository: `TjieLee/R2V_DATA_V2`
- Active development branch: `feature/h3-audio-binding-v1`
- Parent branch: `feature/v3-boogu-reference-edit`
- Production code baseline: `f4cdec251095ba3fd70c57f0a4082c58e5a67101`
- Branch point: `60aaa3877f8e928e5d7eb309950a406d070d4344`
- Current scaffold HEAD: the Git commit containing this document; verify with
  `git rev-parse HEAD`.

The branch point contains documentation changes after the production code
baseline. Existing V3 production behavior remains frozen at the baseline above.

## Current Production

- Run: `prod5000-20260809-171855`
- Status: running
- Rule: do not modify, rerun, invalidate, migrate, or write audit output into
  this run.

## Frozen V3 Behavior

- Coverage is at least 7 of 10 sampled frames.
- Maximum candidates per entity is 3.
- Candidate crop padding is 0.08.
- Reference prefilter mode is `conservative_v1`.
- Final background guard mode is `qwen_v1`.
- Scale-collapse source fallback guard mode is `qwen_v1`.
- Background guard failure only unbinds the background.
- `same_parent_fallback_enabled` is currently false.

## Current Development Goal

Build a MiniMax-H3-compatible audio-to-visual-entity binding sidecar. V1 is
audio-only and prioritizes high-precision, visible, single-speaker evidence.
Pose, expression, camera control, model training, offscreen identity propagation,
cross-clip speaker identity, and production-stage integration are out of scope.

## Current Pass

The architectural scaffold review-fix pass is implemented on this branch:

- source full-audio provenance is separate from explicitly requested H3 assets;
- canonical task components distinguish reference generation, audio reference,
  audio reuse, and intentional combinations;
- face tracks retain at most 32 sampled geometry observations for mask
  association without extending V3 `clip.json`;
- ASD visible-face coverage is explicit and incomplete coverage is ambiguous;
- deterministic fusion precedes voice-reference extraction and entity binding;
- strict voice-reference, subject, and full-audio asset invariants;
- pluggable audio, face-track, association, ASD, and evidence interfaces;
- deterministic high-precision fusion;
- deterministic H3 asset numbering and rendering;
- read-only completed-run sidecar CLI using precomputed/fake evidence;
- GPU-free unit tests and V3 compatibility checks.

New implementation is isolated under `r2v_data_v2/h3/` with one CLI,
`tools/build_v3_h3_audio_binding_sidecar.py`. It is not present in
`run_pipeline_v3.STAGE_ORDER`, does not extend `ClipRecord`, and writes only to a
separate atomically published sidecar root.

No existing V3 quality gate, config schema, pipeline stage, `ClipRecord`, export,
or production artifact may change in this pass.

## Tested Commands

Completed local validation for this pass:

```bash
python -m pytest tests/test_h3_audio_binding.py -q
# 26 passed

python -m pytest -q
# 1512 passed, 1 existing Pillow deprecation warning

python -m ruff check .
# All checks passed

git diff --check
# passed
```

## Open Questions

- How does LR-ASD behave on the project's visible-person, overlap, offscreen,
  profile-face, small-face, and dubbed-audio cases relative to Light-ASD and
  TalkNet?
- Which face detector/tracker adapter best preserves stable association with
  existing SAM3 entity IDs?
- What server-measured speech-duration, synchronization, and voice-quality
  thresholds should replace scaffold policy values?
- Should a later version add diarization for offscreen identity propagation?

## Exact Next Task

Review and approve the corrected V1 contracts. After approval, implement a
read-only LR-ASD adapter evaluation on a small isolated sample, with Light-ASD
and TalkNet as baselines. Do not integrate any model into the production V3
stage order until server evidence validates the currently provisional
thresholds.
