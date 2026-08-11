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
  |-- FaceTrackingBackend -> bounded face-track summaries
  |-- EntityFaceAssociationBackend -> face_track_id to entity_id evidence
  `-- ActiveSpeakerBackend -> interval-level face speaking probabilities
          |
          `-- deterministic fusion -> AudioEntityBinding
                  |
                  `-- deterministic H3 IR and rendering
```

No multimodal LLM owns binding, eligibility, task type, asset numbering, or H3
rendering. Real model adapters remain outside deterministic business logic.

## Pretrained ASD Candidates

No backend is selected or installed in this pass. Initial adapter evaluation may
compare these separately maintained research implementations:

- [TalkNet](https://github.com/TaoRuijie/TalkNet_ASD), an audio-visual ASD model
  with released code;
- [Light-ASD](https://github.com/Junhua-Liao/Light-ASD), a lightweight ASD model
  with released code and pretrained weights;
- [TS-TalkNet](https://github.com/Jiang-Yidi/TS-TalkNet), a later extension that
  also consumes a pre-enrolled speaker reference and is therefore more relevant
  to a future identity-aware pass than the initial V1 bootstrap.

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
interval summaries, not frame-by-frame ASD timelines. Audio paths are provenance
pointers; the scaffold does not extract or copy media.

## Schema Decisions

`AudioTrackMetadata` records source/full-audio paths, health, duration, channel
layout, sample rate, and bounded quality evidence.

`FaceTrack` stores a stable ID, temporal extent, sample count, and aggregate
detection confidence. `EntityFaceAssociation` links one face track to one
existing V3 entity with method and confidence.

`ActiveSpeakerInterval` stores a non-overlapping speech interval, bounded
speaking probabilities by face track, synchronization plausibility, and audio
usability. It does not assign an entity.

`AudioEntityBinding` stores deterministic status:

- `bound`: one high-confidence active face maps to one entity;
- `overlap`: at least two visible face tracks are confidently active;
- `offscreen`: speech exists but no visible face has meaningful ASD evidence;
- `ambiguous`: evidence is meaningful but not decisive or association is absent;
- `no_speech`: the interval contains no speech.

`VoiceReference` is published only from a clean bound interval for the same
entity. `H3AudioBindingIR` separates picture assets, semantic subjects, audio
assets, and interval bindings.

## Deterministic V1 Fusion

The scaffold policy exposes explicit thresholds. A bound result requires:

1. audio status `ready`;
2. speech present and interval duration above the configured minimum;
3. exactly one face probability at or above the active threshold;
4. sufficient margin over the second score;
5. one association for that face track above the association threshold;
6. usable audio and plausible synchronization.

Two or more active faces produce `overlap`. No meaningful visible-face score
produces `offscreen`. Intermediate scores, close scores, missing association, or
failed quality/synchronization evidence produce `ambiguous`. The system never
forces every speech interval onto an entity.

Thresholds in the scaffold are interface defaults for deterministic tests, not
validated production values. Server evidence must justify any frozen values.

## H3-Compatible IR and Rendering

Picture assets follow V3 pairing order: `picture_1`, `picture_2`, and so on.
Semantic subjects use the same order: `subject_1`, `subject_2`, and so on. Voice
references are numbered by subject order, followed by the preserved full-audio
asset. Task type is selected programmatically.

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

The evidence file is a strict list of per-clip `AudioBindingEvidence` records.
Missing evidence or backend exceptions become per-clip failed sidecars while
neighboring clips continue.

## Extension Boundaries

Future adapters may add a real audio preprocessor, face detector/tracker,
face-to-SAM entity associator, ASD model, and voice-reference extractor. Later
versions may add diarization, speaker embeddings, offscreen propagation,
cross-clip identity, pose/expression control, or shot-aware H3 tasks without
changing the V1 evidence and binding separation.
