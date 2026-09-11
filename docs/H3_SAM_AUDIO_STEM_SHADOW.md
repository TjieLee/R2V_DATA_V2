# H3 SAM Audio Three-Stem Shadow

This is an opt-in, additive experiment. It never replaces or writes the current
canonical Audio, DiariZen, Qwen3-ASR, MiMo, H3, primary-voice, or Visual
production stages. For the current server paths, dependency boundaries, and
copy-paste launch commands, use `docs/H3_AUDIO_SERVER_RUNBOOK.md`; this document
owns the semantic and artifact contracts.

All generated artifacts live under the legacy/default root:

```text
<audio-production-root>/sam_audio_stem_shadow_v1/
  separation/
  diarization/
  asr/
  mimo_reconcile_stemtext_final_av_markerpolish_v1/
  references/
```

or, for an explicit named pilot:

```text
<audio-production-root>/sam_audio_stem_shadow_v1/runs/<shadow-run-id>/
  separation/
  auk_speech_v1/
  resolved_stems_v1/
  diarization/
  asr/
  mimo_reconcile_stemtext_final_av_markerpolish_v1/
  references/
```

Each directory is an independently owned atomic stage. In particular,
`separation --overwrite` replaces only `separation/`; it cannot remove a
previously published downstream stage. Every downstream consumer verifies the
current upstream byte lineage before a model call: new named-run DiariZen binds the resolved
records hash, ASR binds the DiariZen provenance hash, and reconcile verifies
the same resolved -> DiariZen -> ASR chain. The model-free resolver binds both
SAM and AuK inventories, records and canonical waveform hashes.
Stem-facts are not a dependency. An overwritten
separation therefore makes older downstream stages explicitly stale; they are
never deleted or silently reused.

## Fixed resolved source contract

New named runs require SAM `voice_first`, then the independent AuK Base worker,
then `tools/resolve_h3_audio_stems.py`. Resolution never invokes a model or writes
waveforms. It publishes `resolved_stems_v1` manifests with speech=AuK and
music/SFX=SAM. Clip order, original target audio/hash/frame extent and zero-offset
timeline must agree. AuK failure makes that clip unavailable; there is no SAM
speech fallback, mixing or ranking. SAM speech remains available for standalone
A/B QA only.

DiariZen and Qwen3-ASR consume resolved speech; MiMo speaker snippets and speaker
reuse use the same AuK PCM. MiMo's final audio turn and music reuse use resolved
SAM music/SFX. Full-audio reuse remains the original canonical full audio.
Original AV remains identity/factual authority, with unchanged speaker ownership,
multiple-speaker exclusions, exact intervals and 3/4-call MiMo architecture.

Resolved inventory/record/summary use independent `.1` schemas. New resolved
DiariZen/ASR provenance uses `.5`, reconcile summary `.17`, and finalization
summary `.2`. Existing SAM-only lineage remains readable for already-published
runs; new named-run inference CLIs require the resolved stage and fail if absent.
Audio asset/product shapes and materializer/prompt versions are unchanged:
their existing source-record fingerprints now bind the resolved contract.

## Authority

The original target audiovisual clip remains final factual authority. Resolved AuK
speech and SAM music/SFX stems are auxiliary evidence and may
become sources for explicitly requested stem-native reference assets. A
separator target label is not semantic ground truth and cannot override a
contradiction in the original AV.

Historical production artifacts remain unchanged. Their frozen versions were:

- MiMo prompt `h3_mimo25_unified_av_reconcile_v22`
- MiMo authority `h3_mimo25_av_authority_contract_v16`
- annotation `r2v.h3.mimo25_av_annotation.13`
- backend `r2v.h3.mimo25_backend.24`
- materializer `h3_mimo25_materializer_v16`

Final H3 remains exactly six sections: subject definitions, summary, retention
analysis, detailed description, overall soundscape, and non-diegetic music.
There is no seventh stem or music-timeline section. The new first response has no timed music-event inventory; missing timing cannot
be invented to make an optional reference variant available.

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
   implementation on the speech stem. The outer shadow CLI runs with
   `R2V_PYTHON` and launches the isolated `QWEN3_ASR_ENV/bin/python` persistent
   worker internally. ASR remains authoritative for text and language and is
   never rewritten by MiMo.
4. `run_h3_mimo25_stem_reconcile_shadow.py` reads separation, DiariZen and ASR
   directly. First, music/SFX canonical stems each receive the same v33
   positive-only role-blind audio-only description request concurrently (max two threads).
   Temperature 0, disabled thinking, 1024 tokens, no ASR/reference/ICL/caption.
   After both finish/fail, Turn 1 receives the original video and frozen reference
   images, with official ICL v4 and visual provenance/windows only. SGLang uses
   `use_audio_in_video=false`; no ASR, clusters, bindings or auxiliary candidates
   enter this visual-only request. It returns `MimoVisualDraft`: segment views,
   one-sentence Subject definitions, short retention statements, style opening
   and one full `shot1_visual_description`. The latter is reused verbatim as
   final `visual_blocks[0].text` (block ID `v1`), not independently generated.
   Turn 2 literally extends those messages with the raw visual assistant response
   and authoritative speech/ASR/diarization/binding facts, with no auxiliary
   candidates. It uses `use_audio_in_video=true` and returns
   `MimoSpeechAVAssemblyDraft`: acoustic observation, grounding, warnings, and a
   visual-plus-exact-dialogue caption. All non-dialogue audio is deferred.
   Turn 3 extends the exact Turn 2 prefix with its raw assistant response and
   a short text-only finalization task containing the music/SFX candidates.
   No video, images, ASR or reference facts are reattached; embedded audio stays
   enabled. `MimoAudioFinalizeDraft` contains only summary, caption, soundscape
   and music. Original AV is primary authority; separator descriptions are
   hints, not mandatory truth. Continuous ambience belongs in soundscape,
   audience-only score in music, and localized synchronized sounds or in-scene
   music in the caption. Route each sound once, without evidence-count gates
   or deterministic audio rewriting.
   Explicit ownership combines the three drafts into unchanged annotation .20:
   Turn 1 owns visual observation, definitions, retention and style opening;
   Turn 2 owns acoustic observation, grounding and warnings;
   Turn 3 owns final summary, caption and sound fields.
   Turn 2 may insert dialogue/Sx with local grammatical changes; Turn 3 may
   change non-dialogue audio wording only. Neither may fabricate visual support.
   Prefix/KV cache reuse is optional: cached-token counts are diagnostics only.
   Normal counts: audio=2, visual=1, AV=2, text=0, total=5. Only Sx projection
   issues may trigger unchanged media-free marker polish v4 (text=1, total=6).
   There is no repair, retry, fallback or AV recheck. A failed turn stops that
   clip and preserves completed raw/diagnostics. Auxiliary errors remain
   available and do not prevent the three-turn attempt.
5. `export_h3_sam_audio_stem_references.py` crops an explicitly requested
   reference from its canonical stem. It never selects an interval on a stem and
   then cuts bytes from the original mix. With `--primary-voice-root`, it reuses
   the already accepted V1 primary-voice turn and policy provenance, then cuts
   the same interval from the canonical speech stem. It does not create a new
   eligibility rule, and entities without an accepted primary voice remain
   without a shadow voice reference. Explicit manual manifests remain available
   for music/SFX review references and cannot claim primary-voice eligibility.

Every command accepts or inherits an explicit case manifest. When a manifest is
provided, its ordered clip inventory must match upstream stages. Reconcile may
select an ordered subset of the current named separation inventory; unselected
clips are not counted as skipped or failed.
The ordered `clip_uids` provenance is carried through every stage rather than
reconstructed from sorted mapping keys. The same explicit `--sam-route` is
validated across separation, DiariZen, ASR, and reconcile. A per-clip
separation failure is published as an explicit skip while remaining clips
continue in that order. Shadow DiariZen's `clip_results.jsonl` is hashed and its
`ready`, `empty`, and `failed` states are carried through ASR and
reconcile provenance. A failed DiariZen clip is an explicit upstream failure,
never an inferred empty-speech clip or a reason to fall back to production
DiariZen. No missing/failed historical facts record can block this new path.
Case order and the exact current separation root must match before MiMo calls.
Existing `run_h3_mimo25_stem_facts_shadow.py`, facts and views remain historical
inspection tools only; the new path neither calls nor requires them.

Final raw caption and sound fields remain visible for parseable failed AV
responses. Both auxiliary candidates/errors remain available regardless of AV
success. Failed AV cannot authorize unsafe identity products. Multi-speaker
exclusions remain. The materializer leaves dialogue intact and reads final
sound fields directly from annotation; no timing is invented.
Prompt `h3_mimo25_speech_assembly_v41`, finalizer
`h3_mimo25_audio_finalize_v1`, visual `h3_mimo25_visual_only_v2`, backend .50,
reconcile record .13 / summary .15 and QA data .7 identify this contract.
The review-only record/summary/QA versions change to identify the new raw fields
and three-turn counting contract. Annotation .20, materializer v23, authority v17,
stem policy v6, official ICL v4 and marker polish v4 are unchanged.
Records/QA preserve `visual_raw_response`, `speech_av_raw_response` and
`audio_finalize_raw_response` separately, with `target_video_visual_only`,
`target_video_speech_assembly` and `target_video_audio_finalize` diagnostics,
including partial failed requests. No new audio validator or prose rewriting
is added.

No real inference was run for this patch. Fake clients establish request shape
and bookkeeping only. Embedded audio tokens of zero/missing are warnings, never
a reason to resend AV. Old v29/v30 outputs remain untouched. The 833 server
smoke and human review are still required; no sound-quality improvement is claimed.

Every optional `--output-root` is constrained to the selected versioned shadow
root; it cannot target current `audio/`, `diarization/`, `asr/`, `h3/`, MiMo
outputs, the legacy/default shadow while a named run is selected, or another
named run.

## Named Pilots

Without `--shadow-run-id`, the tools use the default root under
`$AUDIO_PRODUCTION_ROOT/sam_audio_stem_shadow_v1/`. The validated complete
one-clip smoke remains there; no migration, copy, or overwrite is needed.

For an independent pilot, pass the same `--shadow-run-id random10-v1` to every
stage. Its root is
`$AUDIO_PRODUCTION_ROOT/sam_audio_stem_shadow_v1/runs/random10-v1/`, containing
`separation/`, `auk_speech_v1/`, `resolved_stems_v1/`, `diarization/`, `asr/`,
`mimo_reconcile_stemtext_final_av_markerpolish_v1/`, and optional `references/`.
Old `mimo_stem_facts/` and `mimo_reconcile/` outputs are left untouched.
Run IDs must match `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` exactly. Invalid IDs and
symlink redirects are rejected, not normalized. Custom `--output-root` values
must stay within the selected run. Downstream readers use that run's standard
stage paths, so a custom output location is not automatically discovered.
Stored absolute paths and hashes must close over the same run: copied artifacts
whose provenance still points to the default run or another run are rejected.
Case-manifest order is preserved exactly; run naming performs no sampling.

The following Bash commands launch a new pilot using the existing ordered
10-clip `CASE_MANIFEST` and already configured runtime environments. They are
operator commands, not evidence of a new model run. SAM-only Python dependencies
are supplied only to the separation invocation; the remaining stage CLIs run
with `R2V_PYTHON`. The shadow ASR CLI then launches the isolated
`QWEN3_ASR_ENV/bin/python` worker itself. Do not install one stage's packages
into another environment to resolve an import error.

```bash
RUN_ARGS=(
  --audio-production-root "$AUDIO_PRODUCTION_ROOT"
  --shadow-run-id random10-v1
  --case-manifest "$CASE_MANIFEST"
  --sam-route voice_first
)

PYTHONPATH="$SAM_AUDIO_RUNTIME_PYTHONPATH" \
"$R2V_PYTHON" tools/run_h3_sam_audio_stem_shadow.py "${RUN_ARGS[@]}" \
  --sam-audio-code-root "$SAM_AUDIO_CODE_ROOT" \
  --sam-audio-model-path "$SAM_AUDIO_MODEL_PATH" \
  --sam-audio-model-name "$SAM_AUDIO_MODEL_NAME" \
  --sam-audio-t5-base-path "$SAM_AUDIO_T5_BASE_PATH" \
  --sam-reranking-candidates 1

"$R2V_PYTHON" tools/run_h3_auk_speech_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --shadow-run-id random10-v1 --case-manifest "$CASE_MANIFEST" \
  --auk-python "$AUK_PYTHON" --auk-code-root "$AUK_CODE_ROOT" \
  --auk-checkpoint "$AUK_CHECKPOINT" --auk-qwen-path "$AUK_QWEN_PATH"

"$R2V_PYTHON" tools/resolve_h3_audio_stems.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" --shadow-run-id random10-v1

"$R2V_PYTHON" tools/run_h3_stem_diarization_shadow.py "${RUN_ARGS[@]}" \
  --allow-unverified

"$R2V_PYTHON" tools/run_h3_stem_qwen3_asr_shadow.py "${RUN_ARGS[@]}" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --allow-unverified

"$R2V_PYTHON" tools/run_h3_mimo25_stem_reconcile_shadow.py "${RUN_ARGS[@]}" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --model mimo-v2.5 \
  --base-url http://127.0.0.1:8092/v1 \
  --max-completion-tokens 32768 \
  --allow-unverified

"$R2V_PYTHON" tools/finalize_h3_audio_reuse_shadow.py "${RUN_ARGS[@]}" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --allow-unverified
```

The currently validated server model is `facebook/sam-audio-large-tv`; the exact
path and dependency overlay are recorded in `docs/H3_AUDIO_SERVER_RUNBOOK.md`.
The model identifier is derived from the local SAM checkpoint path/config; an
explicit `--sam-audio-model-name` must agree with it. Do not add `--overwrite`
when launching an independent named pilot. Future pilots can use a different
valid ID without touching this run or the default smoke.

## Example Default Pilot

The legacy/default root remains supported for reproducing or inspecting the
validated smoke. New case inventories should normally use a named run instead
of overwriting this root.

```bash
SHADOW="$AUDIO_PRODUCTION_ROOT/sam_audio_stem_shadow_v1"

PYTHONPATH="$SAM_AUDIO_RUNTIME_PYTHONPATH" \
"$R2V_PYTHON" tools/run_h3_sam_audio_stem_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --sam-audio-code-root "$SAM_AUDIO_CODE_ROOT" \
  --sam-audio-model-path "$SAM_AUDIO_MODEL_PATH" \
  --sam-audio-model-name "$SAM_AUDIO_MODEL_NAME" \
  --sam-audio-t5-base-path "$SAM_AUDIO_T5_BASE_PATH" \
  --sam-route music_first \
  --sam-reranking-candidates 1 \
  --case-manifest "$CASE_MANIFEST"

"$R2V_PYTHON" tools/run_h3_stem_diarization_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --sam-route music_first \
  --case-manifest "$CASE_MANIFEST" \
  --allow-unverified

"$R2V_PYTHON" tools/run_h3_stem_qwen3_asr_shadow.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --sam-route music_first \
  --case-manifest "$CASE_MANIFEST" \
  --allow-unverified

"$R2V_PYTHON" tools/run_h3_mimo25_stem_reconcile_shadow.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --case-manifest "$CASE_MANIFEST" \
  --sam-route music_first \
  --model mimo-v2.5 \
  --base-url http://127.0.0.1:8092/v1 \
  --max-completion-tokens 32768 \
  --allow-unverified

"$R2V_PYTHON" tools/export_h3_sam_audio_stem_references.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --primary-voice-root "$AUDIO_PRODUCTION_ROOT/primary_voice" \
  --sam-route music_first \
  --case-manifest "$CASE_MANIFEST" \
  --allow-unverified
```

The static QA builder now reads the new reconcile output directly, without facts.
Use the exact build/serve commands in `H3_AUDIO_SERVER_RUNBOOK.md`; an independent
QA output root preserves older review exports. It shows both raw stages, errors,
original AV, production comparison and final six-section H3 when safe.

Use `--dry-run` to validate inventory selection without constructing a model
backend. Production migration requires a later explicit decision after shadow
distribution and human review.

## RA2VA no-LR-ASD shadow

This is an explicit target-side pilot, not the production default. The default
`--binding-evidence-mode legacy_lr_asd` remains available for rollback with its
existing readers, payloads and directories. Cross-pair is out of scope.

`--binding-evidence-mode none` keeps the frozen Visual reference graph and reads
raw stem DiariZen segments plus exact Qwen3-ASR facts. No current speaker/entity
proposal is supplied. Neutral prior fields are unavailable evidence, not measured
LR-ASD negatives. Original target AV remains MiMo's final speaker-binding
authority. MiMo retains visual, speech/AV, optional acoustic profile and audio
finalize stages; claiming `lr_asd_support` or `lr_asd_conflict` fails closed.

No-binding base jobs use MiMo inventory `.6`, sourced directly from frozen Visual
production and canonical target Audio, without production `h3/samples.jsonl`.
Canonical shadow samples retain Visual order, ownership, instruction and sample
identity; they have no subject voices or inferred entity bindings. Prepared
publication fills exact raw-DiariZen/Qwen3-ASR speech segments under its own
`h3/samples.jsonl` only. No in-pair or cross-pair samples are created. The inventory's
`source_h3_samples_sha256` hashes the deterministic constructed shadow sample
serialization (and the published samples after preparation), not production H3.
Legacy `.4` and previous no-binding `.5` inventories remain readable with unchanged
fingerprints. Use a fresh reconcile/prepared namespace for the new source contract;
old reconcile jobs are not silently reinterpreted. No-binding DiariZen/ASR provenance uses `.6`; legacy
`.4`/`.5` retain their old serialization. The new reconcile stage also publishes
`source_contract.json` containing the explicit mode and fingerprinted jobs.
The acoustic DiariZen implementation may emit neutral bound/cluster compatibility
files, but this mode neither reads nor hashes them as dependencies.

The commands below are intended server commands, not a Codex inference run.
Reuse the existing validated runtime environment variables from the server
runbook. `SHADOW_RUN_ID` identifies an existing resolved AuK/SAM run and
`CASE_MANIFEST` matches that run's ordered inventory. Use fresh output roots;
do not add `--overwrite` to replace the validated legacy stages.

```bash
SHADOW="$AUDIO_PRODUCTION_ROOT/sam_audio_stem_shadow_v1/runs/$SHADOW_RUN_ID"
DIARI_NONE="$SHADOW/diarization_no_lrasd_v1"
ASR_NONE="$SHADOW/asr_no_lrasd_v1"
MIMO_NONE="$SHADOW/mimo_reconcile_no_lrasd_v1"

"$R2V_PYTHON" tools/run_h3_stem_diarization_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --shadow-run-id "$SHADOW_RUN_ID" --case-manifest "$CASE_MANIFEST" \
  --binding-evidence-mode none --output-root "$DIARI_NONE" --allow-unverified

"$R2V_PYTHON" tools/run_h3_stem_qwen3_asr_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --shadow-run-id "$SHADOW_RUN_ID" --case-manifest "$CASE_MANIFEST" \
  --diarization-root "$DIARI_NONE" --output-root "$ASR_NONE" --allow-unverified

"$R2V_PYTHON" tools/run_h3_mimo25_stem_reconcile_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --shadow-run-id "$SHADOW_RUN_ID" --case-manifest "$CASE_MANIFEST" \
  --binding-evidence-mode none --stem-diarization-root "$DIARI_NONE" \
  --stem-asr-root "$ASR_NONE" --output-root "$MIMO_NONE" \
  --base-url "$MIMO_BASE_URL" --model "$MIMO_MODEL" --allow-unverified

"$R2V_PYTHON" tools/prepare_h3_audio_reuse_sources.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" --stem-shadow-root "$SHADOW" \
  --base-reconcile-root "$MIMO_NONE" \
  --prepared-root "$SHADOW/prepared_no_lrasd_v1" \
  --binding-evidence-mode none --stem-diarization-root "$DIARI_NONE" \
  --stem-asr-root "$ASR_NONE"
```

`--allow-unverified` remains the explicit pilot opt-in for unverified stems.
The first three commands support `--dry-run` for source-only preflight. Prepared
publication is model-free and preserves the unchanged Audio-reuse semantics.
`--source-h3-root` is required only by legacy preparation and is not read in
no-binding mode. Production `h3/` is never created or modified by this adapter.
Rollback selects the old roots and `legacy_lr_asd`; it does not rewrite artifacts.
