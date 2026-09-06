# H3 SAM Audio Three-Stem Shadow

This is an opt-in, additive experiment. It never replaces or writes the current
canonical Audio, DiariZen, Qwen3-ASR, MiMo, H3, primary-voice, or Visual
production stages. All generated artifacts live under:

```text
<audio-production-root>/sam_audio_stem_shadow_v1/
  separation/
  diarization/
  asr/
  mimo_stem_facts/
  mimo_reconcile/
  references/
```

Each directory is an independently owned atomic stage. In particular,
`separation --overwrite` replaces only `separation/`; it cannot remove a
previously published downstream stage. Every downstream consumer verifies the
current upstream byte lineage before a model call: DiariZen binds the separation
records hash, ASR binds the DiariZen provenance hash, and facts/reconcile verify
the complete separation -> DiariZen -> ASR -> facts chain. An overwritten
separation therefore makes older downstream stages explicitly stale; they are
never deleted or silently reused.

## Authority

The original target audiovisual clip remains final factual authority. SAM Audio
speech, music, and SFX/ambience stems are separated auxiliary evidence and may
become sources for explicitly requested stem-native reference assets. A
separator target label is not semantic ground truth and cannot override a
contradiction in the original AV.

The shadow keeps the frozen production contracts unchanged:

- MiMo prompt `h3_mimo25_unified_av_reconcile_v22`
- MiMo authority `h3_mimo25_av_authority_contract_v16`
- annotation `r2v.h3.mimo25_av_annotation.13`
- backend `r2v.h3.mimo25_backend.24`
- materializer `h3_mimo25_materializer_v16`

Final H3 remains exactly six sections: subject definitions, summary, retention
analysis, detailed description, overall soundscape, and non-diegetic music.
There is no seventh stem or music-timeline section. Verified music timing is
rendered naturally in the appropriate existing section.

## Separation

The default bounded cascade follows a qforge-inspired strategy without copying
any project-specific implementation:

```text
full canonical clip --"music soundtrack"--> music + residual
residual            --"human voices"------> speech + final SFX residual
```

The optional `voice_first` route reverses the first two prompts. Both routes are
exactly two calls; SFX is never recursively separated. The official adapter
constructs `SAMAudioProcessor` with only the local model path, while
`SAMAudio.from_pretrained` is explicitly local-only. Batch-one target and
residual outputs are read from the first item of their official per-item lists.
The initial text-only pilot passes `visual_ranker=None`, `text_ranker=None`, and
`span_predictor=None`, with `predict_spans=False` and exactly one reranking
candidate, so optional ImageBind, CLAP, Judge, and PE-A-Frame components are not
loaded. Values greater than one are rejected before model construction because
candidate reranking is unavailable under this local-only runtime contract.

Runtime configuration requires local implementation, SAM checkpoint/config,
and T5-base paths. The SAM text-encoder configuration is copied in memory and
only its `name` is redirected to the local T5 path; checkpoint configuration is
never edited. Required files and SHA-256 fingerprints are validated before
model construction and recorded in inventory provenance. The backend forces
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; there is no network fallback.
An explicitly supplied model identifier must exactly match a non-empty
`_name_or_path` in the local SAM configuration. When that field is absent, a
full identifier such as `facebook/sam-audio-small-tv` or
`facebook/sam-audio-large-tv` is accepted only when its basename exactly matches
the local checkpoint directory; the full supplied identifier remains in
provenance. With no supplied identifier, the directory basename is recorded.

Each call records input, target, and residual hashes, prompts, route, model
fingerprint, timing, and any stable candidate/Judge/CLAP metadata exposed by the
runtime. When such quality evidence is unavailable, the state is `unverified`;
the pipeline does not synthesize a score. Every downstream consumer, including
stem-native reference export, requires the explicit pilot-only
`--allow-unverified` flag before consuming such output. Raw model outputs and
the first-pass residual are retained exactly.

Canonical companions are derived only from each mono raw stem and are 32 kHz
dual-mono stereo PCM16 WAV. The channel policy is recorded as
`dual_mono_from_sam_mono_v1`; this shadow does not claim stereo reconstruction.
Raw and canonical durations are independently allowed to differ from the
original target by at most 0.10 seconds under
`zero_offset_bounded_duration_drift_v1`. The actual extents and signed deltas
are retained, drift above 0.05 seconds is marked with an alignment warning, and
the offset remains exactly zero. There is no time stretching or synthetic tail
padding. The original target duration remains semantic timeline authority;
downstream intervals are also bounded by the actual available stem extent, so a
codec tail cannot publish facts past target EOF and a shorter stem records its
uncovered tail.

## Shadow Stages

1. `run_h3_sam_audio_stem_shadow.py` separates complete canonical clips.
2. `run_h3_stem_diarization_shadow.py` runs the existing DiariZen implementation
   on the canonical speech stem. Production DiariZen continues to read the
   original canonical mix.
3. `run_h3_stem_qwen3_asr_shadow.py` runs the existing exact-segment Qwen3-ASR
   implementation on the speech stem. ASR remains authoritative for text and
   language and is never rewritten by MiMo.
4. `run_h3_mimo25_stem_facts_shadow.py` creates aligned speech, music, and SFX
   AV views and emits factual stem evidence. Speech calls receive the exact
   shadow DiariZen segment inventory and authoritative shadow Qwen3-ASR
   text/language as read-only facts; the response schema has no transcript or
   entity-binding field. The views stream-copy original video and use AAC only
   as an annotation proxy; canonical WAV remains Audio authority. Proxy views
   live inside the facts stage and publish atomically with their records, so a
   failed rerun cannot replace media referenced by the prior stage. Calls store
   raw response text and request diagnostics per stem. A failed clip publishes
   an explicit failed facts record while later clips continue.
5. `run_h3_mimo25_stem_reconcile_shadow.py` sends the original full target AV as
   the model media and supplies stem facts as lower-priority structured
   evidence. Existing speaker fail-closed checks remain active.
6. `export_h3_sam_audio_stem_references.py` crops an explicitly requested
   reference from its canonical stem. It never selects an interval on a stem and
   then cuts bytes from the original mix. With `--primary-voice-root`, it reuses
   the already accepted V1 primary-voice turn and policy provenance, then cuts
   the same interval from the canonical speech stem. It does not create a new
   eligibility rule, and entities without an accepted primary voice remain
   without a shadow voice reference. Explicit manual manifests remain available
   for music/SFX review references and cannot claim primary-voice eligibility.

Every command accepts or inherits an explicit case manifest. When a manifest is
provided, its ordered clip inventory must match the stem inventory exactly.
The ordered `clip_uids` provenance is carried through every stage rather than
reconstructed from sorted mapping keys. The same explicit `--sam-route` is
validated across separation, DiariZen, ASR, facts, and reconcile. A per-clip
separation failure is published as an explicit skip while remaining clips
continue in that order. Shadow DiariZen's `clip_results.jsonl` is hashed and its
`ready`, `empty`, and `failed` states are carried through ASR, facts, and
reconcile provenance. A failed DiariZen clip is an explicit upstream failure,
never an inferred empty-speech clip or a reason to fall back to production
DiariZen. A per-clip facts failure is likewise published and becomes an
explicit reconcile upstream failure; none of these failures aborts usable
clips or disappears from summary provenance. Final reconcile preflight requires
the case-manifest order to equal the current separation inventory and requires
both DiariZen and facts lineage to name that exact owned `separation/` root
before constructing the MiMo backend.
Every optional `--output-root` is constrained to the versioned shadow tree; it
cannot target current `audio/`, `diarization/`, `asr/`, `h3/`, or MiMo output
directories.

## Named Pilots

Without `--shadow-run-id`, all six tools retain the exact default layout under
`$AUDIO_PRODUCTION_ROOT/sam_audio_stem_shadow_v1/`. The validated complete
one-clip smoke remains there; no migration, copy, or overwrite is needed.

For an independent pilot, pass the same `--shadow-run-id random10-v1` to every
stage. Its root is
`$AUDIO_PRODUCTION_ROOT/sam_audio_stem_shadow_v1/runs/random10-v1/`, containing
`separation/`, `diarization/`, `asr/`, `mimo_stem_facts/`, `mimo_reconcile/`,
and `references/` (plus the existing owned stem views).
Run IDs must match `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` exactly. Invalid IDs and
symlink redirects are rejected, not normalized. Custom `--output-root` values
must stay within the selected run. Downstream readers use that run's standard
stage paths, so a custom output location is not automatically discovered.
Stored absolute paths and hashes must close over the same run: copied artifacts
whose provenance still points to the default run or another run are rejected.
Case-manifest order is preserved exactly; run naming performs no sampling.

The following Bash commands launch a new pilot using the existing ordered
10-clip `CASE_MANIFEST` and already configured runtime environments. They are
operator commands, not evidence of a new model run. Keep the validated SAM
environment active for separation; keep `R2V_PYTHON` as the main environment
for the remaining stages. The ASR runner still launches the isolated
`QWEN3_ASR_ENV/bin/python` worker; install nothing into the main environment.

```bash
RUN_ARGS=(--audio-production-root "$AUDIO_PRODUCTION_ROOT"
  --shadow-run-id random10-v1 --case-manifest "$CASE_MANIFEST"
  --sam-route music_first)

python tools/run_h3_sam_audio_stem_shadow.py "${RUN_ARGS[@]}" \
  --sam-audio-code-root "$SAM_AUDIO_CODE_ROOT" \
  --sam-audio-model-path "$SAM_AUDIO_MODEL_PATH" \
  --sam-audio-t5-base-path "$SAM_AUDIO_T5_BASE_PATH" \
  --sam-reranking-candidates 1

"$R2V_PYTHON" tools/run_h3_stem_diarization_shadow.py "${RUN_ARGS[@]}" \
  --allow-unverified

"$R2V_PYTHON" tools/run_h3_stem_qwen3_asr_shadow.py "${RUN_ARGS[@]}" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" --allow-unverified

"$R2V_PYTHON" tools/run_h3_mimo25_stem_facts_shadow.py "${RUN_ARGS[@]}" \
  --temperature 0.2 --allow-unverified

"$R2V_PYTHON" tools/run_h3_mimo25_stem_reconcile_shadow.py "${RUN_ARGS[@]}" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --max-completion-tokens 32768 --allow-unverified

"$R2V_PYTHON" tools/export_h3_sam_audio_stem_references.py "${RUN_ARGS[@]}" \
  --primary-voice-root "$AUDIO_PRODUCTION_ROOT/primary_voice" --allow-unverified
```

The model identifier is derived from the local SAM checkpoint path/config;
an explicit `--sam-audio-model-name` must agree with it. Do not add
`--overwrite` when launching this independent pilot. Future pilots can use a
different valid ID, without touching this run or the default smoke.

## Example Default Pilot

```bash
SHADOW="$AUDIO_PRODUCTION_ROOT/sam_audio_stem_shadow_v1"

python tools/run_h3_sam_audio_stem_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --sam-audio-code-root "$SAM_AUDIO_CODE_ROOT" \
  --sam-audio-model-path "$SAM_AUDIO_MODEL_PATH" \
  --sam-audio-model-name facebook/sam-audio-small-tv \
  --sam-audio-t5-base-path "$SAM_AUDIO_T5_BASE_PATH" \
  --sam-route music_first \
  --sam-reranking-candidates 1 \
  --case-manifest "$CASE_MANIFEST"

python tools/run_h3_stem_diarization_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --sam-route music_first \
  --case-manifest "$CASE_MANIFEST" \
  --allow-unverified

python tools/run_h3_stem_qwen3_asr_shadow.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --sam-route music_first \
  --case-manifest "$CASE_MANIFEST" \
  --allow-unverified

python tools/run_h3_mimo25_stem_facts_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --sam-route music_first \
  --case-manifest "$CASE_MANIFEST" \
  --temperature 0.2 \
  --allow-unverified

python tools/run_h3_mimo25_stem_reconcile_shadow.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --case-manifest "$CASE_MANIFEST" \
  --sam-route music_first \
  --max-completion-tokens 32768 \
  --allow-unverified

python tools/export_h3_sam_audio_stem_references.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --primary-voice-root "$AUDIO_PRODUCTION_ROOT/primary_voice" \
  --sam-route music_first \
  --case-manifest "$CASE_MANIFEST" \
  --allow-unverified
```

Use `--dry-run` to validate inventory selection without constructing a model
backend. Production migration requires a later explicit decision after shadow
distribution and human review.
