# Target-Side TA2VA Shadow

TA2VA `.2` is the frozen target-side Audio-conditioned downstream consumer of an
**ownership-enabled T2VA v7** run (backend `.7`, no-reference core `.6`). T2VA
itself still publishes exactly three sections and normally makes two calls:
semantic AV, then the unchanged frozen `audio_finalize_v6`. TA2VA does not rerun
either of those calls.

For the current server-side happy path and validated commands, use
`H3_T2VA_TA2VA_SERVER_RUNBOOK.md`. This document records the product contract and
implementation invariants.

## Current freeze

Current target-side freeze:

- T2VA prompt: `h3_t2va_joint_av_v7`
- T2VA backend: `r2v.h3.t2va_mimo_backend.7`
- T2VA no-reference core: `r2v.h3.no_reference_av_core.6`
- TA2VA product schema: `r2v.h3.ta2va_shadow.2`
- TA2VA speaker profile: `h3_ta2va_speaker_profile_v2`
- reused R2VA audio finalizer: `h3_mimo25_audio_finalize_v6`

TA2VA currently publishes only:

- `full_audio_reuse`: canonical full target audio, unbound `<Audio 1>`,
  `fully_copy`, zero TA2VA model calls;
- `target_speech_reuse`: one synchronized full-timeline speech Audio per eligible
  frozen Sx, plus optional target music reuse, all `partially_copy`. One audio-only
  profile call covers all selected Sx in the clip.

There are no Pictures, Subjects, entities, frame modes, cross-pair, target-voice
reference or standalone music products in the current target-side TA2VA path.
Cross-pair remains deferred to GitHub Issue #13.

## Ownership and sample placement

The loader revalidates the named resolved-stems/DiariZen/ASR lineage and every
T2VA source hash. Raw segment clip/ID/cluster/times and ASR identity must match
the frozen T2VA job/core; raw and ASR source samples must agree. AuK speech
identity must match the resolved named-run record exactly. There is no fallback
to legacy SAM speech.

Speech ranges reuse `ReuseSampleRange` and the existing
`source_sample_rational_to_32k_round_half_even_v1` mapping policy. For each Sx the
materializer allocates a target-duration zero/silence buffer and copies eligible
speech samples back at their original target sample positions. Segments are never
concatenated, compressed, moved or reconstructed from floating-point timestamps.

The source speech stem is the resolved AuK speech stem. Generated speaker reuse
assets are verified 32 kHz stereo PCM16 FLAC. Cross-Sx overlaps and ranges outside
available target/stem media block the affected Sx rather than publishing an unsafe
identity-specific waveform.

Required per-segment acoustic ownership evidence is internal to T2VA v7. TA2VA
uses the shared `speaker_ownership_reasons()` rule, including the existing narrow
`same_speaker_nonlexical` exception. Any unsafe segment blocks its frozen Sx from
speech reuse without changing Sx assignment, presentation or ASR. Exclusions are
reported as `Sx:segment_id:reason` in `reuse_exclusions` and warning counts,
including clips that publish no target-speech product.

There is no LR-ASD, visibility, duration or voice-quality ranking gate. LR-ASD
remains supporting evidence only. Unsafe/multiple-speaker waveform ownership is a
hard exclusion for Sx-specific speech reuse.

## Audio capacity and music

Audio capacity remains three. Speakers are selected in numeric Sx order and have
priority. Music is appended only when:

1. the frozen T2VA finalizer's stripped `non_diegetic_music` is non-empty;
2. it is not `n/a` or `unknown`, case-insensitively;
3. Audio capacity remains after speaker reuse assets.

Music uses the resolved SAM `music_first` stem and the same bounded-copy /
silence-tail behavior as RA2VA. Stem existence alone does not make music present.
No eligible speech means no `target_speech_reuse` product even if music is
positive. `full_audio_reuse` remains a separate product.

Resolved stems in this shadow flow may still require explicit `--allow-unverified`
for the current named-run pilot data. The flag does not weaken source hash or
lineage validation.

## Reused RA2VA contracts

TA2VA deliberately reuses the existing low-level RA2VA implementation where it
matches the product semantics:

- `probe_canonical_target_frames`
- `_check_stem`
- `ReuseSampleRange`
- `_write_verified`
- `project_audio_relationships`
- `_canonical_audio_definition`
- `_canonical_audio_retention`
- `render_h3_prompt`

The existing top-level reuse builder is not reused because it requires MiMo
gN/entity/reference-graph inputs. `RecaptionReferenceContract` likewise requires
Pictures/Subjects. TA2VA does not fabricate either; its adapter stores real Sx,
Audio contracts and authoritative raw sample ranges directly.

`RecaptionAudioContract` allows optional voice characteristics for
`speaker_speech_reuse`. Canonical wording adds those characteristics only when
present. Existing RA2VA reuse contracts with `voice_characteristics=None` retain
their previous rendering and call behavior.

## Speaker profile contract

TA2VA speaker profile v2 extends the frozen RA2VA acoustic-profile instruction
without changing the RA2VA prompt. The model sees localized AuK speech snippets
for already-fixed Sx targets and may describe only supported acoustic/prosodic
traits such as pitch/register, timbre/texture, cadence/speaking rate,
energy/delivery and audible articulation/breathiness/roughness.

It must not:

- change, merge or split Sx;
- mention its own or another Sx token inside `voice_characteristics`;
- compare one supplied speaker against another;
- copy transcript/dialogue content;
- infer emotion/psychology/personality/intent;
- infer identity, role, profession, nationality or demographics.

Each request supplies exact `required_speaker_groups` and
`required_profile_slots`. SGLang receives a request-specific strict schema with
fixed array length and positional `speaker_group` constants. Xiaomi keeps explicit
slots plus deterministic post-validation. Missing, extra or reordered profiles
fail. There is one profile call per eligible clip, no retry and no fallback
profile. `full_audio_reuse` remains model-free.

## Six-section materialization

TA2VA uses the RA2VA six-section names/order:

1. `subject_definitions`
2. `summary`
3. `retention_analysis`
4. `detailed_description`
5. `overall_soundscape`
6. `non_diegetic_music`

`subject_definitions` contains only canonical Audio definitions; no Picture or
Subject labels are fabricated. `detailed_description` starts from frozen T2VA
`integrated_multimodal_description` and is never model-rewritten. The deterministic
materializer inserts only the canonical first-use Sx Audio relationship and,
when present, the canonical music-reuse relationship.

Exact dialogue payloads/order, Sx, shot order and timestamps remain the frozen
T2VA values and are validated before/after Audio relationship projection.
`overall_soundscape` remains unchanged from frozen T2VA. The underlying music
description remains the frozen T2VA finalizer result; only the canonical music
reuse relationship is prepended when music Audio is actually published.

TA2VA has no visual reference-generation task. Its task prefix is therefore
Audio-only:

```text
summary:
[audio reuse] <frozen T2VA internal summary>
```

The shared Ref2VA/RA2VA `audio_task_prefix()` remains unchanged and may still
render `[reference generation + ...]` for those visual-reference products. TA2VA
does not use that prefix.

The final TA2VA prompt must contain zero `<Picture N>`, `<Subject N>` and
`<Video N>` labels.

## Publication

Use a fresh TA2VA run ID for every materialization contract change. Existing T2VA
v7/core `.6` output can be reused directly; do not rerun semantic AV or the audio
finalizer just because TA2VA changes downstream.

Publication is isolated and atomic under:

```text
<audio-production-root>/ta2va_shadow_v1/runs/<ta2va-run-id>/
  inventory.json
  records.jsonl
  summary.json
  artifacts.json
  raw_profiles/
  assets/
  prompts/
  qa/
```

The inventory binds source T2VA/resolved hashes and profile request policy.
Products record source core identity, Sx profiles, Audio contracts, exact ranges,
warnings and actual model-call count. Inputs are hash-checked before/after
publication. Source files and frozen production outputs remain untouched.

## Validated random20 acceptance

The real random20 TA2VA `.2` acceptance run produced:

```text
clip_count              20
product_count           37
ready_count             37
failed_count            0
model_call_count        17
full_audio_reuse        20
speaker_speech_reuse    20
music_reuse             10
target_speech_reuse     17
```

`f3c9b72d7be7b0fda85d152d` intentionally publishes no target-speech product:
all three S1 segments were excluded for `secondary_vocal_activity`. Its
`full_audio_reuse` product remains ready. This is the expected waveform-ownership
policy, not a failure.

The acceptance run named `target-ta2va-random20-v2` predates the final
deterministic summary-prefix cleanup. Its profile/PCM/product success is still
valid acceptance evidence, but its stored prompt text contains the old
`[reference generation + audio reuse]` prefix. Final publication from the current
branch must use a fresh TA2VA run ID to materialize `[audio reuse]`. T2VA v7 does
not need to be rerun.

## Server commands and manual QA

Use `H3_T2VA_TA2VA_SERVER_RUNBOOK.md` for the current exact environment setup,
MiMo launch/check command, T2VA v7 command, TA2VA `.2` command, result inspection
and QA serving command.

Manual TA2VA QA should verify:

1. each Sx reuse Audio is silent outside its owned intervals and preserves the
   original target timing;
2. voice characteristics match the audible speaker and remain purely acoustic;
3. `<Audio N>` ↔ `(Sx)` relationships are correct and no visual reference labels
   appear;
4. exact ASR, Sx, shots and timestamps are unchanged from T2VA;
5. published music reuse matches actual audience-only BGM and obvious BGM is not
   missed.

The static QA builder validates source/media hashes and displays original video,
canonical full audio, frozen T2VA/ASR/Sx, both TA2VA products, Audio ranges and
players, final six-section prompts and raw profile diagnostics. It does not copy,
symlink or transcode source media.

## Freeze / deferred work

Target-side T2VA v7 + TA2VA `.2` is frozen after the random20 acceptance above.
Do not change it for unrelated cleanup. Cross-pair remains deferred to Issue #13:
future donor construction should use one frozen face-similarity target→donor
mapping and exchange visual face/person reference plus speech/voice reference
consistently, while keeping target dialogue/content/timing unchanged.
