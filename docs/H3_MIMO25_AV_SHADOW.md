# H3 MiMo-V2.5 AV Shadow

This experimental path is additive and read-only with respect to the current JEA
production stages. It writes only `mimo25_av_reconcile_v5/` and
`mimo25_h3_shadow_v5/` under the Audio production root.

The independent `H3_QWEN_SPEECH_PRESENTATION_AB.md` shadow is frozen as a
visual-only contract check. It is not a replacement for MiMo's full audiovisual
speaker reconciliation; MiMo is the final AV authority for this shadow path.

## Authority contract

- DiariZen owns exact speech segment times and sample boundaries.
- Qwen3-ASR owns exact transcript text and language.
- frozen Visual V3 references own entity inventory, order, and image content.
- LR-ASD, source clusters, and current entity bindings are proposals.
- one MiMo-V2.5 joint-AV request returns five ordered internal stages:
  `visual_observation`, `audio_observation`, `av_grounding`, `h3_semantics`, and
  `h3_projection`. These separate visual presence, acoustic identity, speaker
  grounding, official H3 semantics, and typed playback order without adding
  model calls.
- the deterministic materializer owns `Sx`, Subject and Audio references, exact
  dialogue, and final H3 formatting.
- post-MiMo voice recovery may create a real target voice asset only from a
  validated clean single-speaker, resolved `visible_entity` / `onscreen_spoken`
  segment for an entity that has no existing target voice. It uses the exact
  DiariZen 32 kHz sample interval in canonical stereo FLAC; LR-ASD, association,
  current binding, and direct-anchor support are not recovery gates.

Every DiariZen segment receives one Stage A `segment_views` row, one Stage B
audio `segment_decisions` row, and one Stage C `segment_groundings` row,
including empty or otherwise non-transcribed segments. Stage A records only
visible entities and exact-window visibility/orientation/face/mouth/articulation
observations. Stage B owns clip-local `gN`, vocal composition, acoustic
resolution, segment delivery, secondary vocal activity, voice profiles, and
non-speech Audio. Stage C may map that exact Stage B group to one frozen `eN` or
null and classify presentation. Only Qwen3-ASR segments with
`asr_status="transcribed"` receive typed `speech` entries in Stage E.

Every Stage A shot contains at least one non-empty visual-only `vN` block. Every
Stage E shot contains a typed reference to at least one of its own visual blocks,
and all visual blocks, transcribed speech segments, and timeline-eligible Audio
events must appear exactly once in authoritative order. A duration-aware hard
floor of 40 visual words below 3 seconds, 80 words from 3 through 8 seconds, and
120 words above 8 seconds catches catastrophic collapse; it is not a target
length and must not induce invented filler. The pipeline expands typed
visual/speech/event parts, exact dialogue, and final syntax. Internal stage
names, `vN`, `gN`, `eN`, `aeN`, segment IDs, evidence codes, and model-authored
H3 placeholders cannot enter final prose.

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
Subject descriptions reject narrow speaker-profile vocabulary. Voice profiles
remain acoustic-first; supported audible age/gender descriptors are allowed,
while nationality, role, named identity, relationship, or personality claims
remain invalid. The full-AV recheck receives the same mapping and exact
transcribed segment inventory.

Stage B also carries one nullable `speaker_voice_profiles` row for each
resolved speaker group that owns authoritative transcribed speech. A non-null
profile is limited to stable audible pitch register, timbre, texture, cadence,
articulation, and genuinely supported accent or dialect; it cannot copy
dialogue or infer nationality, identity, role, relationship, or personality.
The deterministic
materializer uses this profile in the definition and retention prose for a real
voice Audio asset. Within each real shot it cites that Audio relationship only
on the speaker's first speech event; every authoritative segment and `(Sx)` / `<d>`
block remains present, and a later real shot may cite the asset once again.

MiMo never splits or deletes a DiariZen segment. Multiple vocal events can mark
a segment as requiring acoustic refinement, but the authoritative segment and
Qwen3-ASR text remain present. The default inventory scope is
`canonical_visual_target_inventory` with `canonical_wide_coverage=true` and
exact Visual, canonical Audio, DiariZen, Final H3, and segment-evidence coverage.
An explicit case manifest switches the scope to `explicit_case_subset` and sets
canonical-wide coverage false.

Speaker identity, visible-entity binding, and speech presentation are separate
facts. A visible person alone does not establish visible speech. An
`onscreen_spoken` segment may materialize as `<Subject N> (Sx) says, ...` when
Stage A observes speech-correlated articulation and Stage C records
`visible_lip_motion`, or when Stage A confirms visible presence but the
face/mouth is genuinely non-assessable because of a back/profile view,
occlusion, or crop and Stage C records `speaker_visible_mouth_occluded` with AV
alignment or voice continuity. A hidden mouth does not make a visible person
offscreen. Adjacent motion, gaze, expression, breathing, and general body motion
do not establish speech. A visible listener without reliable evidence never
inherits an offscreen speaker.
The same clip-local group may therefore remain `g1` while a later segment becomes
`offscreen` / `entity_id=null` / `offscreen_spoken`.
LR-ASD support, source-cluster support, current bindings, and direct anchors are
proposals rather than visible-speaking proof: MiMo may
restore a supplied entity for an unbound or zero-anchor segment. Every DiariZen
segment enters MiMo regardless of its current LR-ASD/binding evidence.
Before deciding whether a full-AV recheck is necessary, the adapter applies only
semantics-preserving raw canonicalization and fail-closed typed normalization.
Exact duplicate Stage B or Stage C evidence codes are removed in first-occurrence
order, and an exact redundant leading copy of a row's own retention marker is
stripped. A retention phrase that occurs naturally later in the description is
valid prose and is not a structural failure. Unknown Stage A entities are removed
rather than remapped. Unknown or unsupported Stage C bindings are downgraded to
null-entity conservative states, and an explicitly non-visible Stage C binding
cannot retain an `entity_id`. Identity-bearing voice-profile prose is replaced
only with `null`.
Visible-speaker normalization uses the same Stage-A-aware grounded-evidence
predicate as validation: explicit `offscreen_audio` becomes `offscreen` /
`offscreen_spoken`; otherwise an unsupported claim becomes
`no_reliable_entity` / `uncertain`. Malformed offscreen claims never gain
fabricated `offscreen_audio`. The speaker group and other valid evidence are
preserved, `insufficient_evidence` is recorded when capacity permits, and the
complete validator runs after normalization. A clean primary response therefore
publishes without recheck; remaining semantic failures receive the existing one
full-AV recheck, followed by the same normalization and full validation.
Model-authored Subject visual prose containing Audio-profile content is not
rewritten by these normalizers and remains a recheck or fail-closed condition.
`offscreen_spoken`, `voice_over`,
`message_voice_over`, and `device_playback` preserve the authoritative speech
and speaker group but never create a visible mouth-speaking action. An uncertain
presentation similarly removes visible-entity binding rather than guessing.
The clip-local `primary_speaker_group` represents speaker identity rather than
a speech turn. Segment boundaries, pauses, language changes, or ASR changes do
not create a new group by themselves. Resolved segments bound to the same
visible entity reuse one group, while one resolved group cannot map to multiple
visible entities; full-AV recheck must revisit the AV evidence rather than
blindly merging a conflicting assignment.

Each clip must have exactly one `pair_type="canonical"` Final H3 sample. That
sample is the target-observation representative and uses the `visual_only`
conditioning contract. Optional `in_pair` and `cross_pair` variants must carry
identical target video, full audio, Visual references, instruction, and
full-clip Audio semantics; all variant sample IDs remain in provenance, but
they do not create additional MiMo model jobs.

Prompt, policy, annotation schema, and materializer versions are:

- `h3_mimo25_unified_av_reconcile_v20`
- `h3_mimo25_av_authority_contract_v14`
- `r2v.h3.mimo25_av_annotation.13`
- `r2v.h3.mimo25_backend.20`
- `h3_mimo25_materializer_v14`
- `h3_mimo25_reference_selection_v1`
- `h3_mimo25_recovered_voice_quality_v1`
- `r2v.h3.mimo25_inventory.4`
- `r2v.h3.mimo25_record.10`
- `r2v.h3.mimo25_summary.10`
- `r2v.h3.mimo25_failure.5`
- `r2v.h3.mimo25_raw_response.5`
- `r2v.h3.mimo25_h3_shadow.11`
- `r2v.h3.mimo25_h3_shadow_summary.12`

The OpenAI-compatible client defaults to the `xiaomi` transport, model
`mimo-v2.5`, video FPS 4, `media_resolution=default`, disabled thinking,
JSON-object output, temperature 0.2, and 16384 completion tokens. Base64 is the
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
Payloads and API keys are never persisted. Explicitly reported zero video
tokens fail closed. If primary embedded-audio tokens are exactly zero, one
request retries with canonical full audio; explicitly reported zero audio
tokens on that request also fail closed. Unavailable usage details produce
warnings and do not trigger a blind retry.

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
the five staged fields are serialized annotation only and never become final
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

On `xiaomi`, the one canonical-audio fallback uses
`{"type":"input_audio","input_audio":{"data":"..."}}`, and every request
sends `extra_body={"thinking":{"type":"disabled"}}`. On `sglang`, every
primary, fallback, and full-AV-recheck request sends HTTP-top-level
`use_audio_in_video=true`, `reasoning_effort="none"`, and
`chat_template_kwargs={"thinking":false,"enable_thinking":false}`. The primary
request remains one complete target MP4 with embedded audio and no separate
Audio item. Only an explicit primary `audio_tokens==0` adds canonical full audio
as `{"type":"audio_url","audio_url":{"url":"..."}}`, while retaining the
target video. MiMo reasoning is intentionally disabled because this dataset
path needs deterministic structured annotation rather than agentic
chain-of-thought, while avoiding unnecessary tokens and latency. Nonzero
reported reasoning tokens are retained as runtime diagnostics. Only `stop` is
an explicitly successful finish reason; unavailable finish reason is retained
as a warning, while token limits and every other explicit non-stop reason fail
closed without a semantic recheck.

The local SGLang transport was validated at `http://127.0.0.1:8092/v1` with
`mimo-v2.5` on 8 H200 GPUs using TP8, DP2, and DP-attention. The observed smoke
reported `video_tokens=28800`, `audio_tokens=53`, `reasoning_tokens=0`, and
`reasoning_content=None`, with the full AV request completing in about 17.1
seconds. That SGLang checkout carries an external runtime patch for upstream
`sglang#37060` (MiMo audio encoder deadlock under TP8+DP2); R2V does not modify
or own that external source patch.

After the authoritative primary or canonical-audio-fallback response is
selected, parse or semantic validation failure may trigger at most one full AV
recheck with the same references, target video, and selected audio modality.
There is no text-only semantic repair. The staged annotation assigns
chronological `ae1`, `ae2`, ... IDs to non-speech Audio events and requires
exact ordered Stage E coverage for every event except audience-only
non-diegetic music. Typed `speech` parts exactly cover authoritative transcribed
segments, and typed `visual` parts exactly cover Stage A blocks in their owning
shots. The deterministic MiMo materializer expands those typed references into
the existing internal Qwen3.8 draft interface, then uses the unchanged
validated final renderer. It does not repair missing, extra, duplicated,
reordered, or misplaced model parts.

Music events distinguish audible in-scene `diegetic_music` from audience-only
`non_diegetic_music`. A typed non-diegetic event requires global music status
`present` and a non-null grounded description; `absent` cannot coexist with such
an event. Diegetic music alone does not imply an audience-only score, while a
continuous global score may be present without a localized event.

`overall_soundscape` remains a core section. Any audible ambience, room tone,
environmental layer, physical sound, or non-verbal human sound requires
`overall_soundscape_status=present` and a concise grounded description. It does
not repeat dialogue/speech, narration/voice-over content, singing, diegetic
music, or non-diegetic music/BGM/score. Diegetic music stays in
`detailed_description`, while audience-only music stays in
`non_diegetic_music`. The
model may use `absent` only for verified silence of this soundscape layer, which
materializes as `N/A`. `unknown` is reserved for genuinely unavailable or
uncertain Audio evidence and fails closed during shadow materialization instead
of silently masquerading as confirmed silence. Visual context may disambiguate
an audible source but can never invent room tone or another sound. MiMo performs
a full-timeline audible-event pass that includes brief physical sounds such as
door actions, knocks, footsteps, impacts, object handling, scraping, clicks,
bells, and beeps. Each grounded event appears once at its typed chronological
timeline position, while `overall_soundscape` may summarize salient events in
natural wording rather than repeating the event sentence mechanically.
Localized object interactions such as one door close, knock, footstep, impact,
placement, scrape, or physical click are `physical`; background sources such as
wind, rain, water, traffic, crowds, and outdoor layers are `environmental`.
Operating machinery/device mechanisms are `mechanical`, device signals are
`electronic`, and non-verbal vocal sounds are `human_non_speech`. The examples
are illustrative rather than exhaustive. `single` denotes one transient,
`repeated` several distinct repetitions, and `continuous` a sustained layer or
operation; a multi-frame event window does not by itself make a brief sound
continuous.

Before recheck, a `present` soundscape that accidentally contains dialogue or
music wording may be rebuilt only from its already published, chronological typed
events in `physical`, `environmental`, `mechanical`, `electronic`,
`human_non_speech`, or `other`. Exact duplicate descriptions are removed;
diegetic and non-diegetic music are excluded. If an eligible event itself contains
dialogue/music semantics, or no clean eligible event exists, the adapter does not
fabricate a replacement and retains the normal recheck/fail-closed path.

The materialized output remains official MiniMax H3 Ref2VA: the six sections
are emitted in `subject_definitions`, `summary`, `retention_analysis`,
`detailed_description`, `overall_soundscape`, `non_diegetic_music` order;
reference labels keep one meaning across sections; and dialogue uses stable
`(Sx)` source IDs with `<d>[Language] ...</d>`. The mandatory draft contract
does not redefine official reference-label or retention-marker semantics. The
typed timeline is internal annotation structure only and never appears in the
final Ref2VA text. Approximate event times are internal placement evidence, and
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
errors remain batch-level fail-fast checks. Speaker-source validation uses the
last explicit `<Subject N> (Sx)` or `(Sx)` source attached to each exact dialogue
clause, without a fixed character lookback that could truncate a valid long
voice-reference description.

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

MiMo records retain explicit `present`, `absent`, or `unknown` status for the
overall soundscape and non-diegetic music. The H3 training prompt renders a
soundscape description only for `present`, and renders verified `absent` as
`N/A`. An `unknown` soundscape remains explicit in the annotation but fails
closed during shadow materialization rather than being rendered as silence.
Music retains its independent three-state rendering contract. Neither section
emits prose such as "not established" into training prompts.

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
