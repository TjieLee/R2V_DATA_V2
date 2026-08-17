# R2V_DATA_V2 Project State

Last updated: 2026-08-17

## Repository

- Repository: `TjieLee/R2V_DATA_V2`
- Active development branch: `feature/audio-entity-pairing-h3-v1`
- Visual parent branch: `feature/v3-runtime-integrity-v1`
- Visual baseline at integration: `87bd4e06107d7f56df550979b0e96515cb70f911`
- Audio source branch: `feature/h3-audio-binding-v1`
- Audio source HEAD at integration: `3c4d28018e41461bd296f6e40d985e35a04c6ab5`
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

Run the frozen MiniMax-H3-compatible Audio path over the complete eligible V3
set with explicit, reusable production stages. Pose, expression, camera control,
model training, offscreen identity propagation, global transitive identity
clustering, and Omni remain out of scope.

## Current Pass

The Audio/H3 V1 integration branch is based on the latest Visual integrity
branch and retains the earlier audio scaffold as isolated commits. It adds:

- fixed-root production orchestration for complete Visual occurrence
  enumeration, Audio binding, primary voice, face/speaker embeddings, and
  frozen in/cross pair publication without calibration sampling or HUMAN-label
  dependencies;

- `r2v.audio.clip_binding.1` with deterministic merged speech turns, one final
  Visual reference per occurrence, one primary voice reference per bound
  subject, and normalized face/voice/text embedding asset provenance;
- blockwise NumPy and optional FAISS top-K candidate retrieval;
- strict pairwise same-person and same-voice evidence without transitive
  clustering;
- deterministic in-pair and all-speaking-subject cross-pair construction;
- `r2v.audio.pair_sample.1` and export-relative `r2v.h3.sample.1` contracts;
- draft-only H3 rendering with configurable speech delimiters;
- atomic separate-root publication and a unified `tools/audio_data.py` CLI;
- precomputed, FFmpeg, and external-subprocess adapter boundaries with no base
  model dependency.
- canonical LR-ASD pilot sidecars that feed the binding producer without manual
  directory copying, plus explicit selected/ready/ineligible/failed accounting;
- post-merge voice eligibility for continuous 25 FPS LR-ASD evidence, unique
  transcript-segment assignment, and optimal one-to-one multi-subject donors;
- explicit draft `<Subject N>` / `<Picture N>` / `<Audio N>` bindings.

The face and speaker models remain evidence adapters rather than standalone
acceptance policy. Real server pilots validated Audio binding and froze the
primary-voice and pair policies; this Mac implementation pass does not rerun
models, GPUs, or production data. See `docs/AUDIO_ENTITY_PAIRING_H3_V1.md` for
the canonical contract.

### Earlier audio scaffold

The bounded LR-ASD pilot infrastructure is implemented on this branch:

- source full-audio provenance is separate from explicitly requested H3 assets;
- canonical task components distinguish reference generation, audio reference,
  audio reuse, and intentional combinations;
- face tracks retain at most 32 sampled geometry observations for mask
  association without extending V3 `clip.json`;
- ASD visible-face coverage is explicit and incomplete coverage is ambiguous;
- deterministic fusion precedes voice-reference extraction and entity binding;
- strict voice-reference, subject, and full-audio asset invariants;
- isolated external LR-ASD subprocess runtime with the official 25 FPS, 16 kHz
  mono, S3FD, shot-aware tracking, crop, and inference path;
- strict LR-ASD-native JSON preserving raw class-1 logits, native `score >= 0`
  decisions, and checkpoint provenance without probability claims;
- independent local Silero VAD subprocess bridge without diarization;
- deterministic 25-FPS-to-V3 timestamp alignment and face-box/entity-mask
  association diagnostics;
- `reference_generation`-only fusion and a self-contained manual review bundle;
- bounded, read-only pilot selection with per-clip failure isolation;
- GPU-free fake-evidence tests and V3 compatibility checks.

The implementation remains isolated under `r2v_data_v2/h3/`. The precomputed
sidecar CLI remains available, and the real pilot entry point is
`tools/eval_h3_audio_binding_lr_asd.py`. Neither is present in
`run_pipeline_v3.STAGE_ORDER`; neither extends `ClipRecord`; both write only to
separate output roots.

No existing V3 quality gate, config schema, pipeline stage, `ClipRecord`, export,
or production artifact may change in this pass.

## Tested Commands

Completed GPU-free validation for this integration pass with Python 3.12.13:

```bash
python -m pytest tests/test_h3_audio_production.py \
  tests/test_h3_audio_binding.py \
  tests/test_h3_primary_voice.py \
  tests/test_h3_embedding_pilot.py \
  tests/test_h3_pairing_pilot.py \
  tests/test_h3_audio_dataset.py -q
# 119 passed

python -m pytest -q
# 1819 passed, 1 existing Pillow deprecation warning

python -m ruff check .
# All checks passed

git diff --check
# passed
```

The repository-local `.venv` uses Python 3.9.6. Its compatible Audio/H3 dataset
subset passed `31 passed, 5 skipped`; the skips and the legacy binding-test
collection limit are caused by frozen Visual/H3 Python 3.10+ syntax and were not
worked around by changing Visual code.

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

Review this bounded infrastructure, then configure isolated LR-ASD and Silero
environments on the server and run a small explicit-clip pilot outside every V3
run root. Inspect the review bundles before comparing Light-ASD or TalkNet.
Do not add audio binding to the production V3 stage order until real evidence
validates the provisional association and fusion policies.
