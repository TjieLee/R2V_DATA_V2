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

Calibration planners may bound a review sample, but their sampling quotas are
not production policy. In particular, `max_clips_per_parent` is confined to the
pair-calibration planner. Production H3 paths retain every eligible occurrence,
allow every eligible occurrence to receive face embeddings, and allow every
occurrence with a valid primary voice reference to receive speaker embeddings.
Parent/source provenance may influence retrieval ordering, never eligibility or
a count quota. Strict identity evidence decides whether a pair is usable;
uncertainty produces no pair rather than truncating the dataset. Production
configuration must not expose `max_clips_per_parent` or call calibration
selection helpers with their sampling limits intact.

PairPolicy calibration may use direct HUMAN SAME labels to exclude all members
of the same connected calibration component from a hard-negative review queue.
That closure is bookkeeping for human review only: it does not create a
production identity cluster or publish implied pairs. HUMAN DIFFERENT labels are
negative calibration evidence, while UNCERTAIN labels contribute to neither
positive nor negative distributions. Offline threshold simulation may report
confusion counts for an explicitly supplied policy, but it must not optimize,
select, or persist a production threshold. Speaker similarity remains supporting
evidence until the reviewed positive and hard-negative distributions justify a
specific policy; a low speaker cosine alone is not currently a hard rejection.

## Evidence-First Architecture

```text
completed V3 run + source video
  |-- existing annotation, pairing, references, frames, and entity masks
  |-- isolated LR-ASD raw-video run -> face tracks + native ASD logits
  |-- independent Silero VAD -> speech/no-speech intervals
  `-- timestamp-aligned face-box/entity-mask association evidence
          |
          `-- deterministic fusion -> AudioEntityBinding
                  |
                  |-- pilot-only review audio and visualization
                  `-- reference_generation-only H3 IR
```

No multimodal LLM owns binding, eligibility, task type, asset numbering, or H3
rendering. No voice-reference asset is published by the pilot.

## Pretrained ASD Candidates

[LR-ASD](https://github.com/Junhua-Liao/LR-ASD) is the first real read-only pilot
backend. Its environment stays isolated from the R2V environment and is invoked
by subprocess. The pilot runs the official raw-video path once per clip:

- convert the model video to 25 FPS;
- extract 16 kHz mono audio;
- detect faces with S3FD;
- run shot-aware IoU face tracking;
- use the official face-crop preprocessing;
- run official LR-ASD inference.

The unmodified vendor output is converted to strict JSON before R2V business
logic reads it. The vendor model returns a native class-1 logit when labels are
absent, and its demo uses `score >= 0` as the active decision. The pilot stores
that raw score and native decision explicitly. It does not describe the score as
a calibrated probability. Independent speech activity uses a local Silero VAD
JIT model on CPU; V1 does not perform speaker diarization.

Baseline comparisons remain:

- [Light-ASD](https://github.com/Junhua-Liao/Light-ASD), a lightweight ASD model
  with released code and pretrained weights;
- [TalkNet](https://github.com/TaoRuijie/TalkNet_ASD), an audio-visual ASD model
  with released code.

TS-TalkNet remains a possible future identity-aware experiment because it uses
a pre-enrolled speaker reference, but it is not a V1 baseline or implementation
dependency.

Before production use, audit license and checkpoint provenance, output timing,
offline loading, server dependency isolation, and behavior on dubbed, offscreen,
overlapping, profile-face, and small-face clips. Oxford `ca-subtitle` and
`av-diarization` remain reference architectures, not dependencies in this pass.

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

The sidecar and pilot read `clip.json` through a strict H3-owned projection,
not the current full Visual `ClipRecord` model. Audio consumes only clip/source
identity, readable source metadata, annotation entities, coverage admission,
pairing retention, and ready entity-reference image paths. Visual-internal
reference-edit, attribute, review, and diagnostic fields are ignored; missing
or inconsistent projected identity and binding fields remain hard failures.

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
face-track IDs, native per-face score evidence, exact ASD coverage ratio,
synchronization plausibility, and audio usability. It does not assign an entity.
Every scored face must be visible, and the ratio must equal scored-visible faces
divided by all visible faces. Both the strict LR-ASD-native artifact and the
normalized H3 evidence preserve the 25-FPS frame-level native logits. Review
audio extraction may merge adjacent deterministic bound states without changing
the machine-readable evidence.

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

The pilot policy exposes explicit, configurable thresholds. Face tracks are
aligned independently to each V3 sampled-frame timestamp by nearest timestamp
within the configured tolerance. Association records timestamp delta, face-box
coverage by each entity mask, face-center containment, matched sampled slots,
temporal consistency, and the top-1/top-2 entity margin. Face-box/mask IoU alone
does not decide association.

A bound result requires:

1. audio status `ready`;
2. speech present and interval duration above the configured minimum;
3. exactly one face with a backend-native active decision;
4. complete ASD coverage for visible faces;
5. one association for that face track above the association threshold;
6. usable audio and plausible synchronization.

Two or more active faces produce `overlap`. `offscreen` requires sufficient ASD
coverage and explicit low visible-speaker evidence. If any visible face is
unscored, the interval is `ambiguous`, even when every available score is low.
Intermediate scores, close scores, missing association, or failed
quality/synchronization evidence also produce `ambiguous`. The system never
forces every speech interval onto an entity.

Unfrozen scaffold thresholds remain diagnostic defaults. Production keeps the
manually reviewed Audio binding semantics, calibrated
`voice_reference_quality_v1`, and HUMAN-validated `h3_pair_policy_v1` unchanged;
this orchestration does not retune any of them.

## LR-ASD Pilot CLI

The independent calibration pilot accepts explicit clip IDs and/or a bounded
limit. It does not appear in `run_pipeline_v3.STAGE_ORDER`, and its output must
be outside the source run:

```bash
export LR_ASD_CODE_ROOT=/path/to/Junhua-Liao/LR-ASD
export LR_ASD_PYTHON=/path/to/lr-asd-venv/bin/python
export LR_ASD_MODEL_PATH=/path/to/pretrain_AVA.model
export SILERO_VAD_PYTHON=/path/to/vad-venv/bin/python
export SILERO_VAD_MODEL_PATH=/path/to/silero_vad.jit

python tools/eval_h3_audio_binding_lr_asd.py \
  --run-root /path/to/completed-v3-run \
  --output-root /path/outside/source-run/h3-lr-asd-pilot \
  --clip-id <clip_uid>
```

The LR-ASD environment is responsible for its official dependencies and
`ffmpeg`. The Silero environment must contain its local package and JIT model;
the bridge never downloads a model. Per-clip failures are written to
`failures.jsonl` and do not stop neighboring clips.

The formal batch path is a separate read-only production orchestration. It
enumerates every eligible Visual subject occurrence without a limit or parent
quota, then reuses the frozen Audio binding, primary voice, InsightFace,
SpeechBrain, and PairPolicy implementations. Its fixed stage order is
`audio -> primary-voice -> embedding -> pair`, under
`$AUDIO_RUN_ROOT/production`. The production inventory is complete rather than
sampled; face inference receives every eligible occurrence, speaker inference
receives only occurrences with a valid primary voice, and absent evidence never
deletes a valid in-pair.

Production keeps occurrence-to-occurrence identity decisions in
`pair_evidence.jsonl`, but publishes `in_pairs.jsonl` and
`cross_pairs.jsonl` as target-clip samples. An in-pair contains every included
speaking subject with its own target picture and primary voice. A cross-pair
keeps the target video, full audio, Audio sidecar, and target pictures, while
substituting only donor primary voices. Multi-speaker cross-pairs require a
complete one-to-one legal assignment. Assignment maximizes total face cosine
and uses occurrence IDs for deterministic ties; voice remains only the frozen
`>= 0.20` contradiction gate. Incomplete assignments retain the clip in-pair
and publish no cross-pair. No transitive clustering is performed. Every selected
subject mapping is included in `pairs/review.html`.
See `docs/SERVER_AUDIO_PILOT_RUNBOOK.md` for exact stage and review commands.

The subsequent Omni semantic milestone is deliberately separate from pair
construction. It reads the final in-pair target inventory, analyzes each unique
target video at most once, and reuses that same target semantic record for any
cross-pair. It cannot change speech-turn identity/timestamps or consume donor
media. See `docs/H3_OMNI_SEMANTIC_AUGMENTATION.md`; final H3 rendering remains
out of scope.

Each successful case produces:

```text
review/<clip_uid>/
  source.mp4
  visualization.mp4
  timeline.json
  audio_binding.json
  lr_asd_native.json
  face_entity_association.json
  bound_audio/*.wav
```

`bound_audio` files are human-review artifacts only. They are not
`VoiceReference` or H3 audio-conditioning assets.

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

Future passes may evaluate the pilot evidence, compare the baseline ASD models,
and add voice-reference extraction only after deterministic fusion. Later
versions may add diarization, speaker embeddings, offscreen propagation,
cross-clip identity, pose/expression control, or shot-aware H3 tasks without
changing the V1 evidence and binding separation.
