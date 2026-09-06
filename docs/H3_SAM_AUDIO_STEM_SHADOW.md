# H3 SAM Audio Three-Stem Shadow

This is an opt-in, additive experiment. It never replaces or writes the current
canonical Audio, DiariZen, Qwen3-ASR, MiMo, H3, primary-voice, or Visual
production stages. All generated artifacts live under:

```text
<audio-production-root>/sam_audio_stem_shadow_v1/
```

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
exactly two calls; SFX is never recursively separated. The official adapter uses
the local-only `SAMAudio` / `SAMAudioProcessor.from_pretrained` API and the
official `model.separate(..., predict_spans=False,
reranking_candidates=N)` target/residual contract. Runtime configuration must
name an existing local implementation and model path; the code never downloads
a checkpoint.

Each call records input, target, and residual hashes, prompts, route, model
fingerprint, timing, and any stable candidate/Judge/CLAP metadata exposed by the
runtime. When such quality evidence is unavailable, the state is `unverified`;
the pipeline does not synthesize a score. Raw model outputs and the first-pass
residual are retained exactly.

Canonical companions are derived only from each raw stem and are 32 kHz stereo
PCM16 WAV. Source, raw, and canonical durations must agree within 0.10 seconds,
which is the existing clip-timeline quantization boundary. No time stretching is
allowed, and every stem retains zero timeline offset.

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
   as an annotation proxy; canonical WAV remains Audio authority.
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
Every optional `--output-root` is constrained to the versioned shadow tree; it
cannot target current `audio/`, `diarization/`, `asr/`, `h3/`, or MiMo output
directories.

## Example Pilot

```bash
SHADOW="$AUDIO_PRODUCTION_ROOT/sam_audio_stem_shadow_v1"

python tools/run_h3_sam_audio_stem_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --sam-audio-code-root "$SAM_AUDIO_CODE_ROOT" \
  --sam-audio-model-path "$SAM_AUDIO_MODEL_PATH" \
  --sam-audio-model-name facebook/sam-audio-large \
  --sam-route music_first \
  --sam-reranking-candidates 1 \
  --case-manifest "$CASE_MANIFEST"

python tools/run_h3_stem_diarization_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --sam-route music_first \
  --case-manifest "$CASE_MANIFEST"

python tools/run_h3_stem_qwen3_asr_shadow.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --case-manifest "$CASE_MANIFEST"

python tools/run_h3_mimo25_stem_facts_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --sam-route music_first \
  --case-manifest "$CASE_MANIFEST"

python tools/run_h3_mimo25_stem_reconcile_shadow.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --case-manifest "$CASE_MANIFEST"

python tools/export_h3_sam_audio_stem_references.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --primary-voice-root "$AUDIO_PRODUCTION_ROOT/primary_voice" \
  --sam-route music_first \
  --case-manifest "$CASE_MANIFEST"
```

Use `--dry-run` to validate inventory selection without constructing a model
backend. Production migration requires a later explicit decision after shadow
distribution and human review.
