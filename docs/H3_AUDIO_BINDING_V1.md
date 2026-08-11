# H3 Audio Binding V1

## Scope

H3 Audio Binding V1 is a read-only sidecar over a completed R2V V3 run. It binds
speech evidence to existing visual `entity_id` values and emits structured,
MiniMax-H3-compatible intermediate data. It does not join the production V3
stage order, modify `clip.json`, rewrite references, or alter V3 quality gates.

V1 is audio-only. Pose, expression, camera control, model training, speaker
diarization, offscreen identity propagation, cross-clip speaker identity, and
shot-aware prompt generation are extension points rather than V1 behavior.

## Precision-First Data Policy

A clean entity-bound interval requires exactly one confidently active visible
speaker, a stable face-track-to-entity association, usable audio, sufficient
speech duration, and plausible audiovisual synchronization.

V1 does not use these as clean entity-bound supervision:

- concurrent visible speakers;
- ambiguous ASD evidence;
- offscreen speech without strong identity evidence;
- missing face-track association;
- badly desynchronized or dubbed material;
- missing, corrupted, or unusable audio;
- speech too short or contaminated for a useful voice reference.

These rules apply only to the new audio-binding sidecar dataset.

## Evidence-First Architecture

```text
completed V3 run + source video
  |-- existing annotation, pairing, references, and entity IDs
  |-- AudioPreprocessorBackend -> bounded audio metadata
  |-- FaceTrackingBackend -> bounded sampled face geometry
  |-- EntityFaceAssociationBackend -> compare face geometry with V3/SAM3 masks
  `-- ActiveSpeakerBackend -> interval-level face speaking probabilities
          |
          `-- deterministic fusion -> AudioEntityBinding
                  |
                  |-- clean bound intervals -> VoiceReferenceBackend
                  `-- explicit task variant -> deterministic H3 IR and rendering
```

No multimodal LLM owns binding, eligibility, task type, asset numbering, or H3
rendering. Real model adapters remain outside deterministic business logic.

## Pretrained ASD Candidates

No backend is installed in this pass. The primary candidate for the first
read-only server evaluation is
[LR-ASD](https://github.com/Junhua-Liao/LR-ASD). It is evaluated as an adapter
candidate, not accepted as production behavior by this document.

Baseline comparisons remain:

- [Light-ASD](https://github.com/Junhua-Liao/Light-ASD), a lightweight ASD model
  with released code and pretrained weights;
- [TalkNet](https://github.com/TaoRuijie/TalkNet_ASD), an audio-visual ASD model
  with released code.

TS-TalkNet remains a possible future identity-aware experiment because it uses
a pre-enrolled speaker reference, but it is not a V1 baseline or implementation
dependency.

Before choosing one, audit its license, checkpoint provenance, face-crop and
audio preprocessing contract, output timing, offline loading, server dependency
isolation, and behavior on dubbed, offscreen, overlapping, profile-face, and
small-face clips. The business layer consumes only normalized interval scores.

## Durable Sidecar Contract

The source run is read-only. A sidecar output uses an independent root:

```text
<output_root>/
  summary.json
  audio_bindings.jsonl
  clips/<clip_uid>/audio_binding.json
```

Publication is atomic. Output must not be inside the source run. V1 stores
interval summaries and at most 32 ordered geometry samples per face track, not
unbounded frame-by-frame timelines. The H3 sidecar does not add fields to
`clip.json`.

`AudioTrackMetadata.full_audio_path` is source evidence only. Merely observing a
full-audio path never publishes an H3 conditioning asset. A full-audio asset is
created only for an explicitly requested `audio_reuse` variant.

## Schema Decisions

`AudioTrackMetadata` records source/full-audio paths, health, duration, channel
layout, sample rate, and bounded quality evidence.

`FaceTrack` stores a stable ID, temporal extent, total detection count, aggregate
detection confidence, and a bounded set of samples containing `frame_index`,
`timestamp`, `bbox_xyxy`, and detection confidence. The association backend also
receives the read-only source run root and `masks.rle.json` path, so a future
adapter can compare sampled face boxes with existing V3/SAM3 entity masks.
`EntityFaceAssociation` records the resulting method and confidence.

`ActiveSpeakerInterval` stores a non-overlapping speech interval, the visible
face-track IDs, bounded speaking probabilities, exact ASD coverage ratio,
synchronization plausibility, and audio usability. It does not assign an entity.
Every scored face must be visible, and the ratio must equal scored-visible faces
divided by all visible faces.

`AudioEntityBinding` stores deterministic status:

- `bound`: one high-confidence active face maps to one entity;
- `overlap`: at least two visible face tracks are confidently active;
- `offscreen`: speech exists but no visible face has meaningful ASD evidence;
- `ambiguous`: evidence is meaningful but not decisive or association is absent;
- `no_speech`: the interval contains no speech.

`VoiceReferenceCandidate` has no `entity_id`. The deterministic binding is
established first; only then may `VoiceReferenceBackend` extract candidates from
clean bound intervals. Code assigns a candidate to the containing bound interval
and publishes at most one `VoiceReference` per eligible subject entity.

`H3AudioBindingIR.task.components` is a canonical ordered list drawn from:

- `reference_generation`;
- `audio_reference`;
- `audio_reuse`.

This replaces the coarse task-type flag. A component must agree exactly with its
assets. A voice-reference entity must exist in `subjects`, V1 voice references
are limited to `reference_type=subject`, and at most one full-audio asset is
allowed.

## Deterministic V1 Fusion

The scaffold policy exposes explicit thresholds. A bound result requires:

1. audio status `ready`;
2. speech present and interval duration above the configured minimum;
3. exactly one face probability at or above the active threshold;
4. sufficient margin over the second score;
5. one association for that face track above the association threshold;
6. usable audio and plausible synchronization.

Two or more active faces produce `overlap`. `offscreen` requires sufficient ASD
coverage and explicit low visible-speaker evidence. If any visible face is
unscored, the interval is `ambiguous`, even when every available score is low.
Intermediate scores, close scores, missing association, or failed
quality/synchronization evidence also produce `ambiguous`. The system never
forces every speech interval onto an entity.

All thresholds in the scaffold are unvalidated interface defaults for
deterministic tests. They are not production values. LR-ASD and baseline server
evidence must justify ASD coverage, active-speaker, association, duration, sync,
and voice-quality thresholds before any adapter or policy is frozen.

## H3-Compatible IR and Rendering

Picture assets follow V3 pairing order: `picture_1`, `picture_2`, and so on.
Semantic subjects use the same order: `subject_1`, `subject_2`, and so on. Voice
references are numbered by eligible subject order. A separate audio-reuse
variant publishes one full-audio asset. The two asset types are combined only
when all corresponding task components were explicitly requested.

The renderer derives text from the structured IR, for example:

```text
<Subject 1> is the described entity in <Picture 1>.
<Audio 1> is the voice-timbre reference for <Subject 1> (S1).

summary:
[reference generation + audio reference] ...
```

Rendered text is never the source of truth.

## Precomputed Evidence CLI

The first-pass CLI consumes a completed V3 run and a strict precomputed evidence
JSON file. This supports fake tests and future model adapters without loading a
GPU model:

```bash
python tools/build_v3_h3_audio_binding_sidecar.py \
  --run-root /path/to/completed-v3-run \
  --evidence-json /path/to/precomputed-evidence.json \
  --output-root /path/outside/source-run
```

The default creates a `reference_generation + audio_reference` variant. A
separate audio-reuse output is explicitly requested and written to a different
sidecar root:

```bash
python tools/build_v3_h3_audio_binding_sidecar.py \
  --run-root /path/to/completed-v3-run \
  --evidence-json /path/to/precomputed-evidence.json \
  --output-root /path/to/audio-reuse-sidecar \
  --task-component reference_generation \
  --task-component audio_reuse
```

Task components must be supplied in canonical order. Adding both
`audio_reference` and `audio_reuse` intentionally requests a combined variant;
it never happens because `full_audio_path` exists.

The evidence file contains one strict `PrecomputedClipEvidence` per clip:
pre-fusion `AudioBindingEvidence` plus separate post-fusion voice extraction
candidates. Missing evidence or backend exceptions become per-clip failed
sidecars while neighboring clips continue.

## Extension Boundaries

Future adapters may add a real audio preprocessor, face detector/tracker,
face-to-SAM entity associator, ASD model, and voice-reference extractor. Later
versions may add diarization, speaker embeddings, offscreen propagation,
cross-clip identity, pose/expression control, or shot-aware H3 tasks without
changing the V1 evidence and binding separation.
