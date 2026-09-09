# H3 Audio Reuse Assets (Phase 1)

This is an additive, model-free library prototype. It is **not wired into H3
publication**, ConditioningVariant, AudioKind, retention markers or slot allocation.
Existing target/cross voice references, music references and full-audio reuse remain
unchanged. Phase 2 requires separate review.

## API

`r2v_data_v2.h3.audio_reuse.build_audio_reuse_assets` takes:

- the authoritative `MimoClipJob` from stem reconcile;
- its finalized `MimoAVAnnotationDraft`;
- the matching `SAMAudioStemRecord`;
- `audio_production_root` and a new, separate `output_root`;
- explicit `allow_unverified=True` for the current unverified SAM pilot outputs.

The output directory must be outside the production root, must not contain source
media, and must not already exist. One invocation builds one clip. The caller
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
secondary vocal activity are reused, without visibility or voice-quality gates.
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

## Phase 2 (Not Implemented)

After review, integrate target speech signal reuse (`partially_copy`), keep donor
voice conditioning as `reference` using existing donor selections, and optionally
add music reuse based on finalized semantics. Allocate at most three Audio slots,
speakers first. Preserve target dialogue, full-audio `fully_copy`, legacy
music-reference behavior and donor-specific profile ownership. Derive the task
prefix from the actual Audio contract and distinguish signal-copy wording from
voice-reference wording. None of these product changes is implemented here.
