# R2V_DATA_V2 Project State

Last updated: 2026-08-18

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

Calibrate DiariZen-assisted clip-local speaker continuity against the frozen
LR-ASD plus Visual identity evidence. The pilot reuses the exact ordered 20
targets already reviewed for ASR, preserves overlapping anonymous speaker
segments, and propagates a sparsely anchored entity to all segments in the same
cluster. Production execution remains blocked until human calibration.

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

Audio binding, primary voice, embeddings, and PairPolicy V1 are complete and
frozen. The active producer now reads `production/pairs/in_pairs.jsonl`, rebuilds
the same canonical frozen bound turns, and sends each exact source-audio turn
crop independently to Whisper-large-v3 with `task=transcribe`. Code owns all
identity, entity, and timestamp fields. Cross-pairs never create extra jobs, and
video, donor media, primary voice, embeddings, and PairPolicy evidence never
reach the ASR backend. See `docs/H3_WHISPER_ASR.md`.

The frozen LR-ASD-derived bound turns are a temporary ASR V1 segmentation baseline,
not a Whisper backend dependency. Turn records carry generic segmentation and
entity-binding provenance, including an optional anonymous speaker-cluster ID.
A future DiariZen inventory may replace the boundaries after clip-level
cluster-to-Visual-entity overlap mapping without changing Whisper inference.
The separate DiariZen-assisted binding pilot now implements that calibration
inventory, raw overlap-preserving schemas, sample-domain sparse identity
anchoring, cluster propagation, summary, and static human review. It does not
modify or rerun ASR V1.

The official DiariZen runtime and current model candidate remain unvalidated on
the server. Real production execution is blocked, and the released model
weights are research/non-commercial CC BY-NC 4.0. See
`docs/H3_DIARIZEN_SPEAKER_BINDING.md`.

The validated server pilot selected 20 of 75 target clips and 82 turns. After
installing the required CUDA runtime wheels, it produced 81 transcribed, one
uncertain, and zero failed records. Human QA labeled all 82 turns: 59 `CORRECT`,
15 `WRONG`, and 8 `UNCERTAIN` (59/74, or 79.7%, among explicit correct/wrong
decisions). Whisper-large-v3 remains the dedicated ASR baseline, while
language/accent, proper-name, short-context, and segmentation errors block any
claim that raw output is final transcript ground truth.

The dots3 native-video runtime remains technically operational, but human QA of
the 20-clip semantic pilot found severe hallucinated dialogue. Existing
`semantic_pilot20` output is diagnostic evidence only. dots3 transcript
generation and complete semantic production are blocked; do not delete or
promote that pilot. The earlier native-video transport contract remains
documented in `docs/H3_OMNI_SEMANTIC_AUGMENTATION.md` for diagnostics.

The validated dedicated runtime uses checkpoint architecture
`Dots3NoteForCausalLM` / model type `dots3_note`, `bfloat16`, unquantized
weights, and vLLM commit `e0e5a7fb2808504ba86c94f7b379e38496002fd0`
(`0.27.2rc1.dev191+ge0e5a7fb2`). Its canonical media contract is shared-root
`file` transport under `/mnt/workspace`; no separate HTTP media server is
required.

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

Completed GPU-free validation for the DiariZen-assisted binding implementation
with Python 3.12.13:

```bash
python -m pytest tests/test_h3_diarization*.py -q
# 16 passed

python -m pytest tests/test_h3_*.py -q
# 205 passed

python -m pytest -q
# 1872 passed, 1 existing Pillow deprecation warning

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

Stage and validate the isolated DiariZen environment and local model cache, then
run the fixed 20-clip pilot and review every cluster mapping. Use the review to
calibrate a production mapping policy before lifting the explicit production
block. ASR V2, enhancement, unresolved-speaker MLLM handling, entity-to-subject
mapping, and final `<d>` rendering remain future work.
