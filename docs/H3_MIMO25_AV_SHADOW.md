# H3 MiMo-V2.5 AV Shadow

This experimental path is additive and read-only with respect to the current JEA
production stages. The current path is the named SAM shadow entry, publishing
`mimo_reconcile_stemtext_final_av_markerpolish_v1/` after existing separation,
DiariZen and ASR. Historical `mimo25_av_reconcile_v5/`, `mimo25_h3_shadow_v5/`,
stem-facts and older reconcile artifacts are not migrated or overwritten.
See `H3_AUDIO_SERVER_RUNBOOK.md` Stage 4 for current run and QA commands.

The independent `H3_QWEN_SPEECH_PRESENTATION_AB.md` shadow is frozen as a
visual-only contract check. It is not a replacement for MiMo's full audiovisual
speaker reconciliation; MiMo is the final AV authority for this shadow path.

## Authority contract

- DiariZen owns exact speech segment times and sample boundaries.
- Qwen3-ASR owns exact transcript text and language.
- frozen Visual V3 references own entity inventory, order, and image content.
- LR-ASD, source clusters, and current entity bindings are proposals.
- four stages run in order, with no standalone music/SFX description calls:
  1. visual-only target video plus frozen images and official ICL produces the
     visual draft; embedded audio is disabled;
  2. original target AV plus the canonical visual draft as text, compact
     entity/Subject mapping and authoritative speech facts produces speaker
     observations/grounding, target-video summary and the dialogue caption;
  3. one pure-audio request profiles all real speaker targets using only compact
     gN/Sx/segment interval labels and exact speech-stem snippets. No AV, images,
     full stems, transcripts or entity/binding context are supplied. This stage
     is skipped when no profile targets exist;
  4. original target AV plus canonical music/SFX stems produces only soundscape
     and audience-only music. It receives no speech stem, snippets or profile
     targets and cannot rewrite dialogue, visuals, speaker labels or summary.
  Stages 2/3/4 use fresh system/user requests without inherited ICL, images or
  assistant history; only stages 2/4 enable embedded target-video audio.
- raw stems are separated views of the same target audio. Coherent music exposed
  by a stem and compatible with the original AV is retained even when quiet in
  the original mix; separator residuals are not automatically factual. There is
  no deterministic music insertion or text-only sound fusion.
- the deterministic boundary preserves frozen Subject/Picture ownership and
  checks generated vocal-event formatting without rewriting dialogue. Adjacent
  same-speaker ASR turns may share one natural `<d>` block; ASR artifacts stay
  unchanged and remain visible in QA. There is no segment-count/order zip.
- post-MiMo voice recovery may create a real target voice asset only from a
  validated clean single-speaker, final `visible_entity` / `onscreen_spoken`
  segment for an entity that has no existing target voice. It uses the exact
  DiariZen 32 kHz sample interval in canonical stereo FLAC; LR-ASD, association,
  current binding, and direct-anchor support are not recovery gates.

Every DiariZen segment receives one Stage A `segment_views` row, one Stage B
audio `segment_decisions` row, and one Stage C `segment_groundings` row,
including empty or otherwise non-transcribed segments. Stage A records only
visible entities and exact-window visibility/orientation/face/mouth/articulation
observations. Stage B owns clip-local `gN`, vocal composition, acoustic
resolution, segment delivery and secondary vocal activity. Stage C may map that
exact Stage B group to one frozen `eN` or null and classify presentation. Supplied transcribed speech belongs in the
caption, preserving dialogue and language. The first dialogue block and speaker
transitions need an explicit known `(Sx)`. Consecutive blocks may omit a repeated
marker only when block count equals chronological speech-fact count and the
adjacent speaker IDs match. Count mismatch is not an inventory failure; it
simply disables continuity inference, so every block then needs a marker.
This is format validation, not proof of speaker identity or segment alignment.
Paired, non-nested dialogue tags with language labels and allowed reference/source
labels are checked; text is never replaced, moved or split by code.
The current pilot remains exactly one shot. Visual observation contains only
`visual_blocks` and `segment_views`, not generated shot boundaries.
The pipeline inserts `"\\n[Shot 1] "` between opening and caption.
Model-authored shot markers or internal placeholders produce issues without retries.
Coarse visual evidence never controls final sentence order or length. There is no visual word-count floor.

The final system block prioritizes caption prose. `style_opening` describes only
global cinematographic style/camera language/lighting, not scene contents or
actions. Model input has one authoritative segment inventory; old
`job.r2v_instruction` prose stays stored for provenance but is not model input.
The four-stage change versions the producer contract, not frozen
reference or speaker authority. This is not a verified quality improvement.

The official Complete Example in `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md`
is unchanged. ICL extracts only its global detailed-description line and complete
Shot 1 body before Shot 2, removes the Shot 1 marker, and wraps the exact prose
as `style_opening` / `shot1_caption` JSON. This is a style subset, not the real
response schema; provenance is `h3_official_ref2va_detailed_shot1_v2`.
See `H3_AUDIO_SERVER_RUNBOOK.md` Stage 4 for current defaults and versions.
Human random10 QA, not automated style checks, determines caption quality.

The compact per-request contract supplies the exact complete segment inventory
and the exact ordered transcribed-segment inventory once. MiMo chooses the
observed shot and temporal position, but it does not derive speech eligibility.
Every Stage C grounding explicitly publishes `entity_id`: an exact supplied ID
for `visible_entity`, otherwise JSON `null`. Each transcribed Stage B decision
also carries its own audible `delivery_style`; every non-transcribed decision
uses `null`.

Each Stage D Subject definition is constrained by a machine-readable, per-request
mapping from its exact frozen `<Subject N>` label to all and only its owning
`<Picture N>` labels. The annotation stores each definition as typed
`subject_label` plus model-authored visual `description`; it stores retention as
typed `subject_label`, constrained `marker`, and visual `description`. The
materializer joins the description with the exact frozen Picture labels as
normal official Ref2VA prose. MiMo does not author Picture provenance, and a
model-authored Picture label is rejected rather than treated as authority.
Subject descriptions reject narrow speaker-profile vocabulary. Turn 3 voice
profiles are acoustic-only: pitch/register, timbre/texture,
cadence/rate and energy/delivery; the prompt forbids identity, profession,
personality, nationality, role and demographics. No full-AV recheck is sent.

Turn 2's intermediate audio schema contains only segment decisions, not voice
profiles. Turn 3 requires a string profile for each non-null primary group
with transcribed speech, using exact speech-stem snippets and compact
group/Sx/segment/time locators only. Entity/Subject/final-binding/presentation
context is never sent to the pure-audio profiler.
It cannot re-decide grouping, binding, presentation, ASR, summary or caption.
The final annotation combines Turn 2 segment decisions with Turn 3 profiles.
Its existing nullable profile contract remains unchanged for legacy compatibility;
only the Turn 3 response item disallows null.
Stage-B uncertain/refinement status does not suppress a real acoustic group.
An unbound/offscreen profile alone never publishes an identity-specific Audio
reference. Recovery still requires a non-null group, final visible entity with
onscreen presentation, single-speaker composition, no secondary activity and
all unchanged duration/RMS/clipping/noise/SNR quality gates.
Profiles remain available for real voice-asset provenance, including MiMo-only
recovered visible speakers with no LR-ASD binding. The existing materializer
maps group to Sx to `RecaptionAudioContract.voice_characteristics`; canonical
Audio definitions retain their `featuring ...` wording. MiMo authors natural
speaker/delivery lead-ins in the direct caption; code never splices fixed speech
clauses into that prose.

MiMo never splits or deletes a DiariZen segment. Multiple vocal events can mark
a segment as requiring acoustic refinement, but the authoritative segment and
Qwen3-ASR text remain present. The default inventory scope is
`canonical_visual_target_inventory` with `canonical_wide_coverage=true` and
exact Visual, canonical Audio, DiariZen, Final H3, and segment-evidence coverage.
An explicit case manifest switches the scope to `explicit_case_subset` and sets
canonical-wide coverage false.

Speaker identity, visible-entity binding, and speech presentation are separate
facts. Final original AV may directly bind a known visible speaker and materialize
an `onscreen_spoken` segment as `<Subject N> (Sx) says, ...`.
LR-ASD, source clusters, lip motion, mouth visibility, temporal alignment,
voice continuity, and Stage B resolved status are supporting clues, not binding
prerequisites. Missing supporting evidence does not clear a plausible binding.
A hidden mouth does not make a visible person offscreen. A visible listener
must not inherit speech when AV explicitly indicates another source.
The same clip-local group may therefore remain `g1` while a later segment becomes
`offscreen` / `entity_id=null` / `offscreen_spoken`.
LR-ASD support, source-cluster support, current bindings, and direct anchors are
proposals rather than visible-speaking proof: MiMo may
restore a supplied entity for an unbound or zero-anchor segment. Every DiariZen
segment enters MiMo regardless of its current LR-ASD/binding evidence.
Before final validation, the adapter retains existing
semantics-preserving raw canonicalization and fail-closed typed normalization.
Exact duplicate Stage B or Stage C evidence codes are removed in first-occurrence
order, and an exact redundant leading copy of a row's own retention marker is
stripped. A retention phrase that occurs naturally later in the description is
valid prose and is not a structural failure. Unknown Stage A entities are removed
rather than remapped. Unknown Stage C bindings are downgraded to
null-entity conservative states, and an explicitly non-visible Stage C binding
cannot retain an `entity_id`. Identity-bearing voice-profile prose is replaced
only with `null`.
The production normalization path no longer calls
`_conservative_visible_speaker_downgrade`. The four evidence-sufficiency issues
`visible_entity_requires_resolved_audio`, `onscreen_grounding_incomplete`,
`visible_entity_requires_confirmed_onscreen_speech`, and
`onscreen_speech_requires_reliable_visible_speaker_evidence` are review-only
diagnostic warnings. They neither clear entity IDs nor block H3 publication.
Stage B single-speaker or ordinary uncertain evidence may receive a final AV
visible binding. Confirmed transcribed overlapping or sequential multi-speaker
speech still blocks identity/H3 publication pending turn refinement.
Malformed offscreen representations retain their existing conservative
normalization and never gain fabricated `offscreen_audio`; absence of supporting
evidence is not converted into offscreen.
Concrete contradictions remain hard: a known entity absent from the segment,
explicit offscreen/voice-over/device
evidence conflicting with an onscreen claim, invalid presentation/entity
combinations, unknown references/speakers, and malformed or wrongly attributed
dialogue. These severity rules do not introduce retries, rechecks, or evidence thresholds.
Backend .39 treats Stage A/C articulation disagreement as a review-only warning
without clearing bindings. Missing Sx is also warning-only when authoritative
speech facts contain exactly one distinct speaker (`direct_single_speaker_marker_missing`).
Caption text is preserved, with no marker insertion. Multi-speaker missing
markers, explicit known-marker mismatch, and unknown speakers remain hard.
Model-authored Subject visual prose containing Audio-profile content is not
rewritten by these normalizers and retains its existing validation diagnostics.
`offscreen_spoken`, `voice_over`,
`message_voice_over`, and `device_playback` preserve the authoritative speech
and speaker group but never create a visible mouth-speaking action. An uncertain
presentation similarly removes visible-entity binding rather than guessing; final
H3 retains model-authored speaker prose and dialogue without
leaking internal uncertainty metadata.
The clip-local `primary_speaker_group` represents speaker identity rather than
a speech turn. Segment boundaries, pauses, language changes, or ASR changes do
not create a new group by themselves. Resolved segments bound to the same
visible entity reuse one group, while one resolved group cannot map to multiple
visible entities. Existing safe same-entity canonicalization remains; unrelated
identity contradictions fail closed rather than receiving a recheck.

Each clip must have exactly one `pair_type="canonical"` Final H3 sample. That
sample is the target-observation representative and uses the `visual_only`
conditioning contract. Optional `in_pair` and `cross_pair` variants must carry
identical target video, full audio, Visual references, instruction, and
full-clip Audio semantics; all variant sample IDs remain in provenance, but
they do not create additional MiMo model jobs.

Prompt, policy, annotation schema, and materializer versions are:

- `h3_mimo25_unified_av_reconcile_v35`
- `h3_mimo25_av_authority_contract_v17`
- `r2v.h3.mimo25_av_annotation.20`
- `r2v.h3.mimo25_backend.40`
- `h3_mimo25_materializer_v23`
- `h3_mimo25_reference_selection_v1`
- `h3_mimo25_recovered_voice_quality_v1`
- `r2v.h3.mimo25_inventory.4`
- `r2v.h3.mimo25_record.11`
- `r2v.h3.mimo25_summary.11`
- `r2v.h3.mimo25_failure.5`
- `r2v.h3.mimo25_raw_response.5`
- `r2v.h3.mimo25_h3_shadow.16`
- `r2v.h3.mimo25_h3_shadow_summary.17`

The OpenAI-compatible client defaults to the `xiaomi` transport, model
`mimo-v2.5`, video FPS 4, `media_resolution=default`, disabled thinking,
JSON-object output, official Ref2VA ICL, temperature 0.0, and 16384 completion tokens. Base64 is the
pilot default. `--transport sglang` is the explicit local-provider alternative
and uses strict `json_schema` constrained decoding with the complete current
`MimoAVAnnotationDraft` schema. Xiaomi remains on its separately validated
`json_object` contract. The base URL never selects transport implicitly;
transport and actual response-format mode are recorded in backend provenance
and its configuration fingerprint.
For SGLang, the full schema is carried only by strict `response_format`; it is
not duplicated in user text. Xiaomi retains the textual schema required by its
JSON-object transport. The Stage B Audio decision uses a
`resolution`-discriminated schema:
`resolved` requires a non-null `gN` `primary_speaker_group`, while acoustic
refinement and uncertain decisions still require the field explicitly and may
publish `null`. This invariant is enforced by SGLang constrained decoding and
again by backend semantic validation.
Payloads and API keys are never persisted. Explicit zero video/image tokens
fail closed without resending media. Zero/missing embedded audio tokens remain
warnings only; no canonical-audio fallback or video resend is performed.
Both the adapter and OpenAI SDK allow one attempt per stage, no hidden retries.
Final AV output is caption plus two final sound strings, with no intermediate
sound description, sound status, absence code or timed-event inventory.
No semantic equivalence, absence, leakage, keyword or scoring gate is added.
`N/A` denotes no eligible content. The same AV validators remain; no recheck.

MiniMax H3 Ref2VA retains its official hard reference limits: at most 9 Images,
3 Audio files, and 12 mixed reference files total. MiMo reference-selection V1
does not relax or bypass those checks. For a frozen Visual sample with more than
9 Pictures, it creates an Audio/H3 task-local projection by repeatedly dropping
one deterministically random hair attribute, then one face attribute, then one
remaining attribute while capacity is still exceeded. It never drops a subject,
object, group, or background reference and fails closed when no attribute can be
removed. Clips already containing at most 9 Pictures remain byte-semantically
unchanged at the selected-reference level.

Selection uses a per-clip PRNG seeded from the selection-policy version and
`clip_uid`; sorted candidates make it independent of process order and Python
hash randomization. Surviving source `<Image N>` identities retain their original
indexes and image IDs in provenance, while their task-local `<Picture N>` labels
are reassigned contiguously. The frozen `r2v_instruction` is not rewritten.
Instead, the machine contract publishes the exact source-Image-to-Picture map
and dropped-attribute provenance. MiMo media input, Subject ownership, shadow
materialization, Audio capacity accounting, and human review all consume that
same serialized projection. Frozen Visual samples and image artifacts remain
unchanged. The limits and label semantics follow the official MiniMax H3
[repository](https://github.com/MiniMax-AI/MiniMax-H3),
[prompt-writing skill](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/SKILL.md),
and [Ref2VA guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/ref-en.txt).
The final six-section prose also remains aligned with the official
[base prompt guide](https://github.com/MiniMax-AI/MiniMax-H3/blob/main/skills/h3-prompt-writing/references/base-en.txt);
the internal observation/grounding fields are serialized annotation only and never become final
headings.

The exact request contract keeps `fps` and `media_resolution` beside the
`video_url` object, not inside it:

```json
{
  "type": "video_url",
  "video_url": {"url": "..."},
  "fps": 4.0,
  "media_resolution": "default"
}
```

Xiaomi retains disabled thinking. SGLang uses strict JSON schema, disabled
thinking and `use_audio_in_video=false` for Turn 1, `true` for Turns 2/4; pure-audio Turn 3 uses `false`.
Turn 3 media order is: minimal acoustic target text, then labeled exact
speech snippets ordered by target and interval. Every transcribed interval for
each real gN/Sx is preserved separately, without best-sample selection. No
transcript, visual media, full stem or binding facts enter this request.
Authoritative source sample indices are mapped to the speech-stem sample rate
with exact rational arithmetic and round-to-nearest (ties-to-even), then cropped
using the existing lossless exact-sample extractor. Runtime-owned temporary FLACs
become data URIs and are cleaned before the same Turn 3 call, including when
source media uses HTTP URLs. No production media is changed.
Turn 4 media order is original target AV, music stem, SFX stem (with compact
media labels). Stem hashes are checked before any model call; invalid media
produces a per-clip failure without mutating upstream data.
Missing/unreported reference-image usage is not an error for Turns 2/4.

With profile targets: visual=1, AV=2, audio-only=1, text=0, total=4. Without
targets the profiling call is skipped, profiles=[], total=3. Optional marker
polish adds exactly one text call (5 or 4 total). No retry/recheck/fallback calls.
Raw fields, in execution order: `visual_raw_response`, `speech_av_raw_response`,
`speaker_profile_raw_response`, `audio_finalize_raw_response`. The profile field
is null when skipped; `raw_responses` excludes that absent slot. Profiling has
its own `speaker_profile_audio_only` diagnostic modality. Legacy standalone-stem description fields are
null in new records; current provenance identifies this request contract.

Current versions: backend .60, visual prompt v4, speech assembly v46,
speaker-profile prompt v2, audio finalizer v6, materializer v26; annotation .20, authority v17, official ICL v4
and speaker polish v4 remain unchanged. V26 records the recovery-eligibility
alignment; the acoustic quality policy remains v1. Materializer v25 resolves each
attribute Subject's owner from the frozen reference contract and explicitly
names that entity Subject in its definition. Background Subjects remain
independent. Actual Audio assets retain the canonical definitions/retention
added in v24; audio presence alone never creates a reference asset.

### Conditional Speaker-Marker Polish

The current backend retains at most one text-only call after parsed/normalized
annotation, using `h3_mimo25_speaker_marker_polish_v4`.
Only `direct_single_speaker_marker_missing`,
`direct_dialogue_speaker_marker_missing`, or
`direct_dialogue_speaker_marker_mismatch` can trigger it, and only without
other hard issues. Unknown speakers/references, malformed dialogue, missing
language, internal syntax, and shot markers are not repaired by polish.
Inputs are the existing caption, authoritative `direct_speech_facts`, and
allowed reference labels. There are no media, binding decisions, or AV rechecks.
Thinking is disabled, temperature is 0, and the text budget is 2048 tokens.

Only Sx additions/removals/replacements are accepted: dialogue contents/order
must match exactly; stripping Sx and normalizing whitespace must leave identical
prose. The candidate is then validated against the same speaker facts.
`needs_review`, request/parse failure, non-marker edits, or unresolved projection
retain the original annotation, caption, and validation outcome, without retry.
Only `h3_semantics.shot1_caption` changes on success.

Clean speech clips cost 1 visual + 2 AV + 1 audio-only = 4 calls; attempted
polish costs 5, including failed text attempts. Subtract one when no profiling
targets exist. `raw_responses` contains the executed main-stage responses. The record separately saves polish
attempted/applied/needs_review/raw_response/error and `text_model_call_count`.
The `speaker_marker_text_only` diagnostic has null image/video/audio token
counts. QA displays polish audit/raw JSON separately from final AV raw.

The local SGLang transport was validated at `http://127.0.0.1:8092/v1` with
`mimo-v2.5` on 8 H200 GPUs using TP8, DP2, and DP-attention. The observed smoke
reported `video_tokens=28800`, `audio_tokens=53`, `reasoning_tokens=0`, and
`reasoning_content=None`, with the full AV request completing in about 17.1
seconds. That SGLang checkout carries an external runtime patch for upstream
`sglang#37060` (MiMo audio encoder deadlock under TP8+DP2); R2V does not modify
or own that external source patch.

The conditional-polish flow has fake-client coverage only. The 833 server smoke
and human QA remain necessary. Old v29/v30 output directories are not migrated
or overwritten. Reconcile record .14, summary .16 and policy v6 distinguish this
request history. Invalid final annotation/caption retains raw AV and auxiliary
evidence; only eligible marker projection issues may use one additional text call.
Later clips continue. Annotation .20, materializer v26, authority v17, and ICL v4
are unchanged; no quality improvement is claimed before real server review.

Identity-product exclusions remain independent of raw caption visibility.
Final sound sections come directly from final AV annotation, without internal
status or stock-prose overrides. Optional music references lacking event timing remain unavailable,
rather than fabricating events or suppressing the base caption.

The materialized output remains official MiniMax H3 Ref2VA: the six sections
are emitted in `subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music` order;
reference labels keep one meaning across sections; and dialogue uses stable
`(Sx)` source IDs with `<d>[Language] ...</d>`. The mandatory draft contract
does not redefine official reference-label or retention-marker semantics. The direct caption contains no internal timeline syntax. Approximate event times are internal placement evidence, and
ordinary observed target-video sounds never create an `<Audio N>` reference;
that label remains reserved for an actual copied or referenced Audio asset.

MiMo summary diagnostics expose original and selected Picture-count
histograms, original/selected reference-kind counts, selected/dropped attribute
types, drop-reason counts, and over-limit clip count. The review page shows the
source `<Image N>` and task-local `<Picture M>` mapping for selected references,
dropped attributes and reasons, Stage A/B/C records, per-segment source and
direct-anchor evidence, presentation/binding-change counts, and final
materialized H3. These diagnostics do not change
`h3_mimo25_reference_selection_v1`: clips at or below nine Pictures remain an
exact no-op, and over-capacity clips retain the existing deterministic
attribute-only trimming policy.

Shadow materialization isolates only final Ref2VA response-contract failures at
the individual sample level. Such a record remains `status="failed"`, carries a
compact `materialization_contract_failed` issue list, and publishes no rendered
prompt, corrected speech, or Audio references; later samples continue normally.
The summary reports both the failed-sample count and validation issue-code
counts. Inventory fingerprints, source H3/ASR provenance, canonical media hashes,
task-local Picture projection, duplicate identities, and schema/programming
errors remain batch-level fail-fast checks. Speaker-source validation requires each expected `(Sx)` in its own dialogue
event's lead-in, not necessarily the closest marker. No pronoun resolution
or fixed speech clause is used.

## Post-MiMo target voice recovery

The model-free shadow materializer preserves every existing ASD-derived target
voice first and considers only missing Subject entities for recovery. A candidate
must be resolved to one `gN`, be `visible_entity` plus `onscreen_spoken`, use
`vocal_composition="single_speaker"`, and have no secondary vocal activity. The
versioned acoustic gate retains the frozen 1.0-second minimum, -40 dBFS minimum
RMS, 0.0001 maximum clipping ratio, required local noise evidence, and 10 dB
minimum estimated SNR. Association confidence and LR-ASD score gates are
intentionally absent. Local noise uses the robust median of 20 ms windows from
the nearby canonical-audio regions outside all authoritative DiariZen speech
segments; less than 0.20 seconds of such context rejects fail closed.

At most one exact segment is selected per missing entity. Existing target voices
keep priority; recovered voices fill canonical Subject order while respecting
Audio <= 3 and Picture + Audio <= 12. An existing in-pair receives a shadow-only
effective voice overlay. If a clip has only a canonical sample and recovery
succeeds, the shadow derives one in-pair while leaving the canonical sample
voice-free. Cross-pairs are never created or modified by recovery. Assets and
machine-readable provenance are published under
`mimo25_h3_shadow_v5/recovered_voice_refs/` and
`recovered_voice_references.jsonl`; no recovery terms enter final H3 prose.

This remains compatible with the official MiniMax H3 Ref2VA contract: every
`<Audio N>` denotes an actual audio signal, Audio numbering is independent of
stable `(Sx)` speaker IDs, no target `<Video N>` is introduced, Subject/Picture
provenance is unchanged, and the final six-section order remains unchanged.
The review page exposes each real Audio asset, Subject/entity, `(Sx)`, source
type, source segment, and browser audio player.

## Optional Audio conditioning variants

The model-free materializer can add two independent shadow variants while
reusing the same clip-level MiMo annotation. Both are opt-in, so default
materialization preserves the existing sample inventory.

- `--enable-full-audio-reuse` derives `<clip_uid>/audio_reuse` from the canonical
  sample. Its sole `<Audio 1>` is the existing canonical 32 kHz stereo lossless
  FLAC, marked `fully_copy`; no file is copied and no production sample is
  modified.
- `--enable-music-reference` derives `<clip_uid>/music_reference` only when a
  structured `non_diegetic_music` event accompanies global music status
  `present`. The event must last at least 1.0 seconds, overlap no authoritative
  DiariZen speech interval, map to a valid rounded 32 kHz range, and pass the
  conservative RMS and clipping checks. At most one event is selected by longer
  duration, lower clipping, stronger usable RMS, earlier time, and event ID.

Music boundaries remain explicitly MiMo-approximate and are recorded as rounded
canonical 32 kHz sample coordinates. The extracted asset is a real stereo FLAC;
it guides qualitative audience-only music style without copying the waveform.
V1 does not publish partial music reuse or treat a dialogue-contaminated mix as
an isolated score. A future policy should consider merging contiguous compatible
non-diegetic-music spans from one cue; V1 deliberately keeps model event spans
independent and does not merge arbitrary adjacent events. Full-audio reuse,
clean music reference, and voice reference
are distinct roles, and Audio numbering remains independent of `(Sx)` numbering.
Every variant keeps all Pictures and is omitted rather than dropping Visual
references if the official Audio <= 3 or Picture + Audio <= 12 limits would be
exceeded. Review UI exposes each role, playable asset, interval provenance, and
music description; Subject/entity/`Sx` metadata appears only for voice assets.

Internal audio statuses remain diagnostic observations. Final H3 uses the
model-authored `h3_semantics.overall_soundscape` and `non_diegetic_music` prose,
not a status-to-stock-sentence mapping. Noncritical soundscape wording issues
are review warnings; an absent/null/false bookkeeping combination cannot alone
invalidate a usable caption.

## Server commands

Run the known-case manifest without calling the API first:

```bash
"$R2V_PYTHON" tools/run_h3_mimo25_av_reconcile.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --known-case-pilot \
  --media-root /mnt/workspace \
  --media-mode base64 \
  --dry-run
```

Run the known cases, then the complete current inventory:

```bash
export MIMO_API_KEY='...'

"$R2V_PYTHON" tools/run_h3_mimo25_av_reconcile.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --known-case-pilot \
  --transport xiaomi \
  --model mimo-v2.5 \
  --fps 4 \
  --media-resolution default \
  --media-root /mnt/workspace \
  --media-mode base64

"$R2V_PYTHON" tools/run_h3_mimo25_av_reconcile.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --transport xiaomi \
  --model mimo-v2.5 \
  --fps 4 \
  --media-resolution default \
  --media-root /mnt/workspace \
  --media-mode base64 \
  --overwrite
```

Run the same MiMo contract against the validated local SGLang endpoint:

```bash
export MIMO_API_KEY=EMPTY

"$R2V_PYTHON" tools/run_h3_mimo25_av_reconcile.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --known-case-pilot \
  --transport sglang \
  --base-url http://127.0.0.1:8092/v1 \
  --model mimo-v2.5 \
  --fps 4 \
  --media-resolution default \
  --media-root /mnt/workspace \
  --media-mode base64
```

Materialize and review without another model call:

```bash
"$R2V_PYTHON" tools/materialize_h3_mimo25_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --overwrite

# Optional inspection variants; both flags default to false.
"$R2V_PYTHON" tools/materialize_h3_mimo25_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --enable-full-audio-reuse \
  --enable-music-reference \
  --overwrite

"$R2V_PYTHON" tools/serve_h3_mimo25_review.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --legacy-qwen38-root "$QWEN38_ROOT" \
  --host 127.0.0.1 \
  --port 8768
```

Open `http://127.0.0.1:8768`. Review annotations are fingerprint-bound and are
stored under `mimo25_h3_shadow_v5/human_review/`.
