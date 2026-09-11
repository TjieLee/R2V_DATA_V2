# Target-Side TA2VA Shadow

TA2VA .1 is an additive downstream consumer of an **ownership-enabled T2VA v7**
run (backend .7, no-reference core .6). Old T2VA runs are not rewritten or given
invented summaries. T2VA itself still publishes exactly three sections and makes
two calls: semantic AV, then the unchanged frozen audio_finalize_v6.

TA2VA publishes only:

- `full_audio_reuse`: original canonical 32 kHz stereo FLAC, unchanged subtype,
  unbound `<Audio 1>`, `fully_copy`, zero model calls.
- `target_speech_reuse`: one synchronized 32 kHz stereo PCM16 FLAC per eligible
  frozen Sx, `partially_copy`. One audio-only profile call covers all selected
  Sx, using localized AuK snippets and frozen speaker_profile_v2. Failed profiles
  fail only this variant; no fallback descriptions or retries.

There are no Pictures, Subjects, entities, frame modes, cross-pair, target-voice
reference or standalone music products. Cross-pair remains deferred to Issue #13.

## Ownership and Samples

The loader revalidates the named resolved-stems/DiariZen/ASR lineage and every
T2VA source hash. Raw segment clip/ID/cluster/times and ASR identity must match
the frozen job/core; raw and ASR source samples must agree. AuK speech identity
must match the current resolved record exactly. No fallback to SAM speech.

Speech ranges use `ReuseSampleRange` and the existing
`source_sample_rational_to_32k_round_half_even_v1` policy. They are copied at
their original positions into zeros of exactly the canonical target frame count.
The source is strict WAV/PCM16/32k/stereo; generated assets are strict
FLAC/PCM16/32k/stereo and decoded-sample verified. No concatenation or float-time
placement. Cross-Sx overlaps and ranges outside available target/stem media make
the affected Sx unavailable, matching the shared-stem isolation constraint.
There is no LR-ASD, visibility, duration or voice-quality ranking gate.

Required per-segment acoustic ownership evidence is internal to the T2VA core.
TA2VA uses the shared `speaker_ownership_reasons()` rule, including its
same-speaker nonlexical exception. Any unsafe segment blocks its frozen Sx from
speech reuse without changing Sx assignments, presentation or ASR. Exclusions
include Sx/segment/reason in summary `reuse_exclusions` and warning counts,
including clips with no published speech variant. A fresh T2VA v7 run is required;
missing evidence in old cores is never inferred as safe.

Speakers are selected in numeric Sx order, up to three. Music is added only when
the frozen T2VA finalizer's stripped music field is non-empty and is neither
`n/a` nor `unknown` (case-insensitive), and capacity remains, matching RA2VA.
Music uses the same bounded copy/silence-tail policy as RA2VA. Omitted speakers
and music have explicit capacity warnings. No eligible speech means no
`target_speech_reuse` product, even when music is positive.

Resolved stems currently remain unverified: explicitly supply
`--allow-unverified` for this shadow pilot, as with the existing reuse path.

## Reused Contracts

The adapter reuses `probe_canonical_target_frames`, `_check_stem`,
`ReuseSampleRange`, `_write_verified`, `project_audio_relationships`,
`_canonical_audio_definition`, `_canonical_audio_retention`,
`audio_task_prefix`, and `render_h3_prompt`.

The existing top-level reuse builder requires MiMo gN/entity jobs, and
`RecaptionReferenceContract` requires Pictures/Subjects. TA2VA deliberately does
not fabricate those inputs: its small product/asset schema stores real Sx and
raw sample ranges. The profile schema likewise uses fixed Sx in speaker_group,
not newly invented gN. It imports the frozen profile prompt, finish diagnostics
and identity-claim check; invalid profiles fail rather than inventing prose.

`RecaptionAudioContract` now allows optional speech-reuse voice characteristics.
Canonical wording appends them only when present. Existing RA2VA contracts with
`None` render exactly as before, with no new inputs or calls.

The six official section names/order are reused unchanged. Subject definitions
contain only Audio definitions. Summary is `audio_task_prefix` plus the frozen
internal summary. Detailed prose is never model-rewritten; only canonical Audio
relationships are inserted. Exact dialogue payloads/order and Sx are checked
before and after injection. Full-audio prose and both sound fields stay unchanged;
speech-reuse music adds only the canonical relation when music Audio is present.

## Small Pilot

First produce a fresh, small T2VA v7 run with the existing source/case selection
CLI in `H3_T2VA_SHADOW.md`. Do not rerun separation/ASR for an already matching
resolved Audio cache. Older cores lack the required structured ownership evidence.

With `T2VA_ROOT` pointing to that completed run:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
export T2VA_ROOT="$AUDIO_PRODUCTION_ROOT/t2va_shadow_v1/runs/ta2va-core-v7-pilot2"

"$R2V_PYTHON" tools/run_h3_ta2va_shadow.py \
  --t2va-root "$T2VA_ROOT" \
  --ta2va-run-id target-ta2va-v1-pilot2 \
  --base-url http://127.0.0.1:8092/v1 \
  --transport sglang --model mimo-v2.5 \
  --media-root /mnt/workspace --allow-unverified --dry-run
```

Remove only `--dry-run` for an explicitly approved real profile pilot. There
is at most one TA2VA model call per clip; full-audio records never duplicate its
accounting. No T2VA AV/finalize or upstream Audio stage is rerun by this command.

Publication requires a fresh run ID and is atomic under:

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
Products record the source core, Sx profiles, Audio contracts, exact ranges and
warnings. All inputs are hash-checked before/after publication. Source files
and production outputs remain untouched.

## Manual Review

```bash
export TA2VA_ROOT="$AUDIO_PRODUCTION_ROOT/ta2va_shadow_v1/runs/target-ta2va-v1-pilot2"
"$R2V_PYTHON" tools/build_h3_ta2va_qa.py \
  --ta2va-root "$TA2VA_ROOT" \
  --media-root /mnt/workspace --media-base-url http://127.0.0.1:8765
python -m http.server 8765 --bind 127.0.0.1 --directory /mnt/workspace
```

Open `qa/review.html` through that server using its path relative to
`/mnt/workspace`. QA validates source/media hashes and displays original video,
full target audio, frozen T2VA/ASR/Sx, both products, Audio traits/ranges/players,
and raw profiles. No media is copied or symlinked into QA. Listen for silence
outside each Sx's ranges, original timing, profile agreement, music presence,
unchanged exact dialogue, and zero visual-reference labels before scaling.

Local tests use synthetic PCM, fake profile completions and offline browser
playback. They do not establish real model/profile quality.
