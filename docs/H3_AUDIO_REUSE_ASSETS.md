# H3 Audio Reuse Assets and Shadow Products

These are additive, model-free libraries. Phase 1 builds verified PCM assets;
Phase 2 consumes them in a separate shadow product materializer. Neither changes
production H3 or reruns MiMo. The legacy materializer entrypoint and its target/cross
voice references, music references and full-audio reuse remain available unchanged.

## API

`r2v_data_v2.h3.audio_reuse.build_audio_reuse_assets` takes:

- the authoritative `MimoClipJob` from stem reconcile;
- its finalized `MimoAVAnnotationDraft`;
- the matching `SAMAudioStemRecord`;
- `audio_production_root` and a new, separate `output_root`;
- explicit `allow_unverified=True` for the current unverified SAM pilot outputs.

The output directory may be in a caller-owned shadow subtree, for example
`<audio-production-root>/sam_audio_stem_shadow_v1/runs/<run-id>/audio_reuse_assets/<clip-uid>/`.
It must not overlap production stages or source media, and must not already exist.
One invocation builds one clip. The caller
owns clip enumeration and output naming. The function returns an
`AudioReuseManifest` and atomically publishes `manifest.json`, `Sx_gN.flac` speaker
tracks and `music.flac`. There is no CLI or production-stage integration.

CPU PCM I/O uses NumPy and the optional `soundfile`/libsndfile package (also used
elsewhere for FLAC review media). It does not load a model, contact an endpoint,
or install dependencies at runtime.

## Contracts

New schemas:

- `r2v.h3.speaker_speech_reuse_asset.1`
- `r2v.h3.music_reuse_asset.1`
- `r2v.h3.audio_reuse_manifest.1`

Speaker assets record real gN plus the existing final `direct_speech_facts` Sx,
ordered source segments, original sample coordinates, mapped 32 kHz coordinates,
job/annotation/stem fingerprints, source/output hashes and actual target frame
count. Optional entity/Subject provenance comes only from a unique finalized
visible-entity binding with single-speaker ownership. Offscreen/unbound speakers
still produce tracks without invented identity. Inconsistent gN/Sx mapping or
multiple entity claims produce exclusions, not a new mapping algorithm.

Every output is PCM16 lossless FLAC, 32 kHz stereo, with the exact decoded canonical
target frame count and zero offset. Source formats and hashes are checked before
copying; inputs are hash-checked again before publication. The output is decoded
and compared sample-for-sample against the expected full PCM array.

Only transcribed, attributable single-speaker intervals are copied into target-sized
digital silence. The existing semantic exclusions for non-single speech and
secondary vocal activity are checked without visibility or voice-quality gates.
Explicit `same_speaker_nonlexical` activity belongs to the same speaker and may
be copied. Different/uncertain secondary speakers remain excluded. Legacy recovered
voice deliberately retains its stricter single-speaker/no-secondary-activity gates.
There is no minimum duration, RMS, SNR, clipping or LR-ASD threshold. Original
sample coordinates map using exact rational arithmetic and round-half-even under
`source_sample_rational_to_32k_round_half_even_v1`. There is no concatenation,
normalization, fade, stretching, synthesis or resampling of copied PCM.

Ambiguous intervals remain upstream but are excluded from reuse. Overlapping
different-speaker intervals (including untranscribed competing speech) cannot be
isolated from the shared stem: affected tracks are excluded. Speech outside the
target or speech-stem extent excludes its whole track; it is never silently
truncated. The manifest lists explicit segment/group reasons. Other usable
speakers and the music asset can still be built.

Music copies from sample zero, truncating only beyond target EOF or adding exact
zeroes for an uncovered target tail. Existing SAM zero-offset bounded drift of
0.10 s is retained. The music asset does **not** assert that music is audible;
`semantic_presence_evaluated=false`. Future publication must consult finalized
Audio semantics, not separator existence.

## Phase 2 Shadow Products

`audio_reuse_materializer.materialize_audio_reuse_products()` reads a prepared,
frozen MiMo `inventory.json` / `records.jsonl`, the matching H3 `samples.jsonl`,
SAM separation `records.jsonl`, and `<reuse-root>/<clip-uid>/manifest.json`.
The prepared root uses the explicit
`r2v.h3.audio_reuse_prepared_source.1` record contract, not legacy
`MimoRecord`. The bridge validates frozen stem-reconcile .14 records with
backend .60 or .61, retaining their original backend/configuration and record
fingerprints. It does not rewrite provenance or migrate annotations.
All selected ready clips, including any donor clips to be used, need matching
manifests. A donor outside the prepared inventory is excluded, never substituted.

Validation covers job/annotation/SAM fingerprints, target media, source intervals,
speaker ownership, entity/Subject ownership, media hashes, decoded frame counts,
32 kHz stereo media, and manifest ordering. Target FLAC subtypes are unrestricted;
SAM canonical stems are PCM16 WAV and reuse outputs are PCM16 FLAC.
No PCM is generated here. Stale
lineage fails before publication. Inputs are rehashed before atomic publication;
existing output directories, source roots and production stages cannot be replaced.

Products use `RecaptionReferenceContract.audios` as the task authority:

| Audio kind | Retention | Ownership |
| --- | --- | --- |
| `speaker_speech_reuse` | `partially_copy` | target Sx; optional paired entity/Subject |
| `cross_voice` | `reference` | target Subject/Sx; full donor speaker track |
| `music_reuse` | `partially_copy` | unbound target music track |
| `full_audio_reuse` | `fully_copy` | unchanged, unbound full target audio |
| `music_reference` | `reference` | existing music-style reference semantics |

In-pair products replace short voice conditioning with target speaker tracks.
When no in-pair exists, a reuse product is derived from the canonical sample.
Canonical visual-only products are retained. Cross products read the existing
`FinalSubjectVoice` donor clip/occurrence mapping, without new matching. Ambiguous
or unavailable donor assets produce explicit omission warnings. Only a uniquely
resolved donor profile may describe donor Audio; target profiles are never used.

Speakers are allocated first in stable Sx order, at most three. Music is last;
capacity omissions are warnings and never remove factual dialogue. Annotation .20
has a final `non_diegetic_music` section rather than a separate music-status enum:
its nonempty non-`N/A`, non-`unknown` content is the frozen positive music decision.
Stem existence alone is never positive evidence. The original music prose is
preserved even if there is no Audio slot. Empty final sections still fail the
existing renderer validation.

Summary prefixes inspect the actual Audio retention types. Mixed donor reference
and target music copy becomes
`[reference generation + audio reference + audio reuse]`; no Audio remains
`[reference generation]`. Signal-copy Audio has no voice-characteristics prose.
Audio relationships are inserted once at each speaker's first verifiable speech
lead-in, without rewriting `<d>` content or existing presentation prose. Music
relationships are added to `non_diegetic_music`, not dialogue. Unlocatable speaker
relationships fail the product rather than guessing. Full-audio reuse wording and
the six-section H3 layout remain unchanged.

The new product contracts are `r2v.h3.audio_reuse_product.1` and
`r2v.h3.audio_reuse_product_summary.1`, with materializer policy
`h3_mimo25_audio_reuse_materializer_v2`. Only this product path projects exact
chronological ASR dialogue into protected frozen caption slots, inserts missing
speech with finalized presentation, and then projects Audio relationships.
Extra or contradictory speaker slots fail closed. Legacy v26 rendering is unchanged.
`ReuseAudioReference` carries the full
multi-segment asset plus manifest path/hash/fingerprint and donor provenance.
This separate versioned product leaves MiMo backend .61, annotation .20, all
prompts and legacy materializer v26 provenance untouched.

## CPU-Only Server Materialization

First prepare the frozen effective input after Phase-1 assets have been produced.
The base defines clip order; an optional override replaces whole records only,
including failures. Unknown/duplicate clips or mismatched job/SAM fingerprints
fail closed. Counts come from the files, never a fixed clip quota.

The bridge reuses the existing production inventory and stem-reconcile job
builders. Prepared H3 keeps every non-speech field, including frozen cross-donor
subject voices, and projects only the exact current transcribed job segments.
Prepared speech has null entity binding; finalized MiMo remains binding authority.
The outer inventory hashes the prepared H3 file without changing job fingerprints.
Both prepare and materialize publish into new owned directories atomically.

```bash
"$R2V_PYTHON" tools/prepare_h3_audio_reuse_sources.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --stem-shadow-root "$SHADOW_RUN_ROOT" \
  --base-reconcile-root "$BASE_RECONCILE_ROOT" \
  --override-reconcile-root "$OVERRIDE_RECONCILE_ROOT" \
  --source-h3-root "$FROZEN_H3_ROOT" \
  --prepared-root "$PREPARED_ROOT"
```

Prepared output: `inventory.json`, `records.jsonl`, `h3/samples.jsonl`,
and `summary.json` (`r2v.h3.audio_reuse_prepared_source_summary.1`).
The summary records input hashes, chosen base/override roots, output hashes,
effective ready/failed counts and zero model calls.

Then materialize. No inference, asset regeneration or production overwrite:

```bash
"$R2V_PYTHON" tools/materialize_h3_audio_reuse_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --mimo-root "$PREPARED_ROOT" \
  --source-h3-root "$PREPARED_ROOT/h3" \
  --separation-root "$SHADOW_RUN_ROOT/separation" \
  --reuse-root "$SHADOW_RUN_ROOT/audio_reuse_assets_v1" \
  --output-root "$SHADOW_RUN_ROOT/h3_audio_reuse_products_v1" \
  --enable-full-audio-reuse
```

Outputs are `records.jsonl` (rendered H3, exact corrected speech and actual Audio
asset paths/provenance) and `summary.json` (input hashes, Audio kinds, task-prefix
counts, failures and capacity warnings). `model_call_count=0` and
`production_artifacts_modified=false`. There is intentionally no overwrite flag.
The legacy music-reference selection/export path remains separate; it is not
renamed to music reuse or rerun by this CLI.
