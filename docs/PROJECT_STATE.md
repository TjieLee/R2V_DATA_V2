# JEA Qwen3 Audio production integration

The active readable JEA Audio production path now runs all seven explicit stages
from canonical Visual `samples.jsonl`: Audio binding, frozen primary voice,
InsightFace/ECAPA embeddings, frozen PairPolicy, DiariZen, isolated local
Qwen/Qwen3-ASR-1.7B, and final H3 rendering. Cross donors remain restricted to
the full media collection path. See `docs/H3_QWEN3_ASR.md`. Legacy Whisper
pilots are retained but are not selected by this production path; Dots3 remains
paused.

# R2V_DATA_V2 Project State

Last updated: 2026-08-19

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

Use a zero-model-call scout over all 75 frozen production Audio/DiariZen targets
to manually choose the non-empty background-rich Target Audio Caption positive
pilot. This dataset has fewer than 20 obvious positives, so its pilot count is
dynamic rather than quota-filled. The deterministic final renderer is frozen;
the scout and both caption pilots do not feed or rebuild it.

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
frozen. The historical ASR V1 producer reads `production/pairs/in_pairs.jsonl`,
rebuilds the same canonical frozen bound turns, and sends each exact source-audio
turn crop independently to Whisper-large-v3 with `task=transcribe`. Code owns
all identity, entity, and timestamp fields. Cross-pairs never create extra jobs,
and video, donor media, primary voice, embeddings, and PairPolicy evidence never
reach the ASR backend. See `docs/H3_WHISPER_ASR.md`.

The frozen LR-ASD-derived bound turns are a temporary ASR V1 segmentation baseline,
not a Whisper backend dependency. Turn records carry generic segmentation and
entity-binding provenance, including an optional anonymous speaker-cluster ID.
The production ASR V2 stage consumes formal DiariZen boundaries after clip-level
cluster-to-Visual-entity overlap mapping without changing Whisper inference.
The DiariZen-assisted binding stage now implements the frozen production
inventory, raw overlap-preserving schemas, sample-domain sparse identity
anchoring, cluster propagation, summary, and pilot-only static human review. It
does not modify or rerun ASR V1.

The official DiariZen pipeline and current model completed the repaired 20/20
server pilot. It produced 50 raw segments and 22 clusters: 21 mapped, one
ambiguous fail-closed abstention, zero unbound, and zero conflict. Human QA
marked all 22 cluster decisions `CORRECT`, with no wrong, uncertain, or
unlabeled cases. The exact threshold-free `h3_diarizen_sparse_anchor_policy_v1`
is now complete and frozen for production. Released model weights remain
research/non-commercial CC BY-NC 4.0. See
`docs/H3_DIARIZEN_SPEAKER_BINDING.md`.

The accepted pilot mapped 114.4155 speaker-seconds: 89.62 direct-anchor seconds
and 24.7955 identity-propagated seconds, approximately 21.7% carried by
within-cluster continuity. Nine manually accepted EOF intersections totaled
0.2245 seconds; maximum overrun was 0.0525 seconds / 840 samples. No numeric
coverage, support-share, or margin threshold is part of the production policy.

Formal DiariZen production is now complete: 75/75 clips ready, 179 raw
segments, and 81 clusters (79 candidate-mapped, one ambiguous, one unbound,
zero conflict). It mapped 461.828 speaker-seconds: 371.345 direct-anchor and
90.483 identity-propagated seconds. Forty EOF-adjusted segments had a 0.0305
second median positive overrun and 0.0805 second / 1288-sample maximum. The 81
production clusters were not all human-reviewed; 22/22 human acceptance applies
only to the calibration pilot.

ASR V2 consumes one exact effective DiariZen segment per Whisper job. It reuses
the exact ordered ASR V1 pilot20 clip IDs for A/B selection, permits nullable
entity identity for unresolved clusters, and transcribes candidate-mapped,
ambiguous, unbound, and conflict segments alike. The accepted pilot contained
50 segments: 48 transcribed, two backend-uncertain, and zero failed. Human QA
labeled all 50 as 41 `CORRECT`, three `WRONG`, six `UNCERTAIN`, and zero
unlabeled. Backend uncertain `2` and human-QA uncertain `6` are separate
concepts. Production inference is now enabled with no transcript-confidence or
text-usability gate. See
`docs/H3_WHISPER_ASR_V2.md`.

The controlled A/B reference used the same 20 clips and model. ASR V1 had 82
LR-ASD-derived units with human QA 59/15/8; ASR V2 had 50 DiariZen units with
human QA 41/3/6. The units differ, so this is not a paired per-turn accuracy
delta. DiariZen-segment ASR V2 is accepted as the production raw-ASR baseline,
not as final H3 transcript truth.

Formal ASR V2 raw production is complete and frozen: 75 clips, 179 segments,
176 transcribed, three backend-uncertain, and zero failed. The model-free
calibration analyzer remains immutable evidence under
`$AUDIO_RUN_ROOT/asr_v2_text_calibration`. Its human-reviewed result selected
the intentionally rounded `0.65` language-probability eligibility threshold.

TextUsabilityPolicy V1 is now complete and frozen. It trusts display text only
when the raw record is transcribed, text is non-empty, and
`language_probability >= 0.65`; otherwise the derived sidecar hides text. The
policy writes every source segment under
`$AUDIO_RUN_ROOT/production/text_usability`, preserves identity fields, and
does not mutate raw ASR, voice, embeddings, or pairs. Language probability is
not transcript correctness probability and identity is not a text gate. The
model-free final renderer consumes this frozen sidecar without recalculating
trust. See `docs/H3_ASR_V2_TEXT_USABILITY.md` and
`docs/H3_FINAL_RENDERER.md`.

Deterministic H3 final renderer V1 is implemented at the fixed
`$AUDIO_RUN_ROOT/production/h3` root. It emits one sample for every frozen
in-pair and cross-pair row, preserves target Visual instruction and target
speech timeline, and changes only voice provenance for cross-pairs. Trusted
subject-bound text is rendered verbatim inside `<d>`; trusted speech without a
pair subject remains structured and unrendered. Language is preserved as
metadata with `language_conditioning_applied=false`. Baseline rendering has
zero model, MLLM, GPU, and parent-quota calls. Formal server dry-run counts and
language distribution remain to be measured on the frozen production root.

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

The first Target Audio Caption MLLM Pilot V1 completed 20/20 runtime calls and
its human-review bundle is available. It reuses the exact ordered ASR V2
pilot20 clip IDs and asks Dots3 only for audible ambience, music, non-speech
events, acoustic style, and cluster-scoped vocal delivery. It sends no
transcript, trusted text, language, entity ID, donor media, or separate audio
item. The clips are predominantly acoustically clean, so this remains
clean/negative evidence for hallucination and abstention behavior; human review
is in progress, and the set cannot validate positive recall or production
readiness.

The current model-free background-audio scout enumerates all 75 production
targets and computes only DiariZen temporal-union coverage and canonical PCM16
RMS/peak diagnostics. It applies no threshold, model, quota, or automatic
selection. The static review bundle under
`$AUDIO_RUN_ROOT/background_audio_scout` exports all manually labeled
`background-rich` clips in source order for the next positive pilot. See
`docs/H3_TARGET_AUDIO_CAPTION.md`.

Current milestone status is therefore:

- **COMPLETE / FROZEN:** DiariZen, ASR V2, TextUsabilityPolicy V1, and the
  deterministic final renderer implementation;
- **CLEAN / NEGATIVE PILOT:** 20/20 runtime complete; formal QA intentionally
  skipped because it lacks useful positive background cases;
- **CURRENT:** manually select the background-rich positive pilot with the
  model-free scout;
- **PENDING:** run the same Target Audio Caption policy on the dynamic selection,
  complete human QA, approve production, integrate with the renderer, and
  rebuild final H3;
- **OPTIONAL LATER:** voice-reference enhancement or denoising experiments.

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

Completed GPU-free validation for the DiariZen-assisted binding, ASR V2 raw
production, model-free text-usability calibration, and the frozen policy with
Python 3.12.13:

```bash
python -m pytest tests/test_h3_asr_v2_transcription.py -q
# 15 passed

python -m pytest tests/test_h3_asr_v2_text_calibration.py -q
# 10 passed

python -m pytest tests/test_h3_text_usability.py -q
# 14 passed

python -m pytest tests/test_h3_asr_transcription.py -q
# 18 passed

python -m pytest tests/test_h3_diarization*.py -q
# 22 passed

python -m pytest tests/test_h3_*.py -q
# 250 passed

python -m pytest -q
# 1917 passed, 1 existing Pillow deprecation warning

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

Implement entity-to-subject mapping and final H3 structured rendering from the
frozen sidecars. Optional enhancement and unresolved-only MLLM handling remain
future work; neither may redefine the frozen raw-ASR or text-usability policy.
