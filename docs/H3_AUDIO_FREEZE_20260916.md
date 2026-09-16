# Audio/H3 frozen state — 2026-09-16

This document is the compact source of truth for the accepted Audio/H3 freeze on
`feature/h3-audio-jea-qwen3-v1`. Historical development sections in
`docs/H3_AUDIO_SERVER_RUNBOOK.md`, `docs/H3_MIMO25_AV_SHADOW.md`, and
`docs/PROJECT_STATE.md` may describe older pilots and version checkpoints; when
they conflict with this file, this freeze note wins.

## Frozen target-side products

T2VA and TA2VA are complete and frozen.

- T2VA prompt: `h3_t2va_joint_av_v7`
- T2VA backend: `r2v.h3.t2va_mimo_backend.7`
- T2VA no-reference core: `r2v.h3.no_reference_av_core.6`
- TA2VA product schema: `r2v.h3.ta2va_shadow.2`
- TA2VA speaker profile: `h3_ta2va_speaker_profile_v2`
- shared audio finalizer: `h3_mimo25_audio_finalize_v6`

The accepted TA2VA random20 run produced 37/37 ready products: 20
`full_audio_reuse`, 17 `target_speech_reuse`, 17 profile calls, and 10 published
music-reuse references. The known `secondary_vocal_activity` case intentionally
publishes no target-speech product.

## Frozen reference-side / RA2VA path

The accepted no-LR-ASD reference-side configuration is:

- speech prompt: `h3_mimo25_speech_assembly_v49`
- speaker profile prompt: `h3_mimo25_speaker_profile_v2`
- audio finalizer: `h3_mimo25_audio_finalize_v6`
- visual prompt: `h3_mimo25_visual_only_v5`
- authority policy: `h3_mimo25_av_authority_contract_v18`
- annotation schema: `r2v.h3.mimo25_av_annotation.20`
- backend: `r2v.h3.mimo25_backend.66`
- materializer: `h3_mimo25_materializer_v29`
- reference selection: `h3_mimo25_reference_selection_v2`
- official ICL: `h3_official_ref2va_detailed_shot1_v4`
- video FPS: `4.0`
- binding mode: `none` (LR-ASD remains supporting evidence only)

Persistent media are served over HTTP to SGLang: target video, frozen reference
images, music stem, and SFX stem. Short request-specific speaker snippets remain
base64. HTTP transport is part of the accepted serving configuration, not a new
semantic policy.

The final HTTP random20 acceptance produced 19/20 ready records, one expected
substantive hard failure, zero text-polish calls, and 77 total model calls.
Production artifacts were not modified. The substantive failure is the preserved
`direct_transcribed_dialogue_missing` hard case; tiny utterances such as `我。`
do not independently invalidate the clip.

The final RA2VA binding A/B review was completed against this HTTP run and the
configuration above is frozen. Do not reopen validator, prompt, binding, or
reference-selection policy for unrelated cleanup.

## Dataloader-facing publication contract

JSONL is the dataset manifest format: one metadata record per line, with media
referenced by validated paths and hashes rather than embedding media bytes. The
current product families are not equally flattened, so dataloaders should not
assume that every top-level `records.jsonl` is sufficient by itself.

### Reference-side R(A)2VA

The primary final Audio-reuse publication is:

```text
<shadow-run>/h3_audio_reuse_products_v1/
  records.jsonl
  summary.json
```

Each `r2v.h3.audio_reuse_product.1` row contains the sample identity, pair type,
conditioning variant, complete Audio reference contracts and provenance,
corrected speech segments, final `rendered_h3_prompt`, task prefix, warnings,
and record fingerprint. This is sufficient for final text/Audio semantics, but
it does **not** embed the frozen reference-image paths or the complete Subject /
Picture source contract. A training dataloader that needs actual Ref2V/RA2VA
reference images therefore still has to join the frozen source H3/Visual
manifest (or an equivalent flattened export). It should not crawl arbitrary
intermediate stage directories, but one file is not yet enough for this family.

For first/last-frame conditioning, the later projection publishes:

```text
<shadow-run>/h3_frame_conditioned_products_v1/
  records.jsonl
  summary.json
  frames/<clip_uid>/{first.png,last.png,metadata.json}
```

Each `r2v.h3.frame_conditioned_product.1` row contains target-video path/hash,
frame-reference paths/hashes, Subjects, Audio references, corrected speech, and
the final rendered prompt. For these derived first/last-frame dataset families,
`records.jsonl` is already a direct dataloader manifest; only the referenced
media files remain external.

### Target-side TA2VA

TA2VA publishes:

```text
<ta2va-run>/
  records.jsonl
  summary.json
  inventory.json
  assets/...
```

Each `r2v.h3.ta2va_shadow.2` row in `records.jsonl` contains the variant, status,
Audio references (including paths/hashes), generated reuse-asset provenance,
speaker profiles, final six-section prompt, warnings, and fingerprints. Because
TA2VA has no visual references by contract, a dataloader can consume final TA2VA
sample metadata from this single `records.jsonl`; the referenced FLAC payloads
remain external files.

### Target-side T2VA

T2VA itself is intentionally less flattened. Its top-level `records.jsonl`
contains status/provenance and hashes, while target-video/source facts live in
`inventory.json` and the semantic core and rendered prompt remain in per-clip
`core/<clip_uid>.json` and `prompts/<clip_uid>.txt`. Therefore `records.jsonl`
alone is not sufficient to reconstruct a ready T2VA sample.

### Recommended dataset export boundary

Do not change the frozen inference producers merely for loader convenience. If
training wants one manifest per family, add a deterministic, model-free export
layer that validates all frozen hashes and emits one row containing all metadata
needed by the loader while continuing to reference image/audio/video payloads by
path and hash.

The export should flatten at least:

- R(A)2VA: final product row + frozen target-video and reference Picture/Subject
  media contracts;
- T2VA: inventory job + validated no-reference core + final three-section prompt;
- TA2VA: the existing product row can be forwarded almost unchanged.

This keeps inference/review artifacts frozen while giving training one stable
JSONL entry point per dataset family.

## Change policy

- Keep frozen T2VA/TA2VA and RA2VA semantic contracts unchanged for unrelated
  cleanup.
- New semantic behavior requires a version bump and a fresh run ID.
- Dataloader convenience should be implemented as a deterministic export layer,
  not by mutating frozen inference artifacts.
- QA and export tooling remain read-only with respect to production inputs.
