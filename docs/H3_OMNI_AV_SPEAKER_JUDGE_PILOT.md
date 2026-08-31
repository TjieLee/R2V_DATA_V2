# H3 Omni AV Speaker Judge Pilot V1

This pilot is a read-only, segment-level Qwen3-Omni observation sidecar. It
does not modify LR-ASD, Audio bindings, DiariZen cluster bindings, ASR, Visual
artifacts, or final H3 samples. Proposed identities are diagnostic only.

Each explicitly selected raw DiariZen segment receives a synchronized context
window from 0.75 seconds before the segment through 0.75 seconds after it,
clipped to the common source boundary. The same absolute boundaries drive both
the neutral review video and the canonical full-audio trim.

The video displays only neutral `eN` or `OTHER` face labels. It never displays
LR-ASD active state, score, current draft binding, DiariZen status, audit flags,
or ASR text. The model receives only the neutral video, canonical trimmed audio,
target interval relative to the window, and visible mapped entity IDs. Human QA
labels may be retained in the pilot manifest but are never included in a model
request.

The V3 observation has two independent axes:

```json
{
  "decision": "visible_entity|offscreen|other_visible|multiple_speakers|uncertain",
  "entity_id": "eN or null",
  "secondary_speech_status": "none|incidental|competing"
}
```

The primary axis answers who safely owns the speech turn. The secondary axis
describes linguistic speech from other speakers. `none` means no meaningful
secondary linguistic speech; non-linguistic vocalizations alone do not count.
`incidental` means another speaker is audible but one primary still clearly owns
the turn. `competing` means meaningful contributions prevent a reliable single
primary. These are semantic distinctions, not duration, word-count, overlap, or
loudness thresholds.

`multiple_speakers` is reserved for materially competing speech where no
reliable primary owns the target turn, and therefore requires
`secondary_speech_status=competing`. Merely hearing more than one voice is not
enough. A clear mapped primary with a brief interjection remains
`visible_entity + incidental`. Multiple offscreen speakers remain
`offscreen + competing` when the important binding fact is that no mapped
visible subject owns the speech.

Pass 1 is blind to the draft. A separately versioned blind Pass 2 runs under the
existing verification conditions and whenever Pass 1 reports `incidental` or
`competing`. Primary-attribution stability compares only `(decision, entity_id)`;
secondary-speech stability compares `secondary_speech_status` independently.
Thus `visible_entity/e2/incidental` followed by `visible_entity/e2/none` retains
the stable proposed entity `e2`, but does not confirm contamination.

The exclusion flags are also independent. A confirmed
`multiple_speakers + competing` has no primary and publishes
`subject_entity_binding_excluded=true`. Confirmed `incidental` or `competing`
secondary speech publishes `identity_specific_voice_products_excluded=true`,
while a clear primary may still retain its entity binding. Unconfirmed one-pass
or pass-disagreement contamination does not set the voice-product exclusion.

A DiariZen segment therefore need not be acoustically single-speaker to retain a
valid primary subject/entity attribution. Identity-specific voice-reference
cleanliness is a separate question. This remains a read-only pilot sidecar with
no production consumer: it does not rewrite primary voice references, speaker
embeddings, identity pairs, bindings, ASR inputs, or final H3 samples.

## Manifest

The input is non-empty JSONL with exact segment identities:

```json
{"schema_version":"r2v.h3.omni_av_speaker_judge_manifest.1","clip_uid":"03e8d3bb9744f7951e545a07","segment_id":"segment_0001","human_label":{"decision":"visible_entity","entity_id":"e1"}}
```

Human labels are optional and are used only by the static review page.

## Positive Pilot

```bash
OLD_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/production/jea_motion_v1/e2e200-random-seed20260821-20260821-143232
POSITIVE_MANIFEST=/tmp/omni-av-positive-03e8.jsonl

printf '%s\n' \
  '{"schema_version":"r2v.h3.omni_av_speaker_judge_manifest.1","clip_uid":"03e8d3bb9744f7951e545a07","segment_id":"segment_0001","human_label":{"decision":"visible_entity","entity_id":"e1"}}' \
  > "$POSITIVE_MANIFEST"

python tools/run_h3_omni_av_speaker_judge.py \
  --audio-production-root "$OLD_ROOT" \
  --case-manifest "$POSITIVE_MANIFEST" \
  --output-root "$OLD_ROOT/omni_av_speaker_judge_pilot_v1" \
  --base-url http://127.0.0.1:8091/v1 \
  --served-model-name Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --checkpoint-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --media-mode file \
  --media-root /mnt/workspace
```

## Random200 Controls

Select every raw segment belonging to the three reviewed clips without using
audit rank or support ratio as a gate:

```bash
NEW_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/random200/random200-src10000-19999-seed20260831-20260831-124415
CONTROL_MANIFEST=/tmp/omni-av-random200-controls.jsonl

jq -c 'select(.target_clip_uid == "a073596149def028cff1305e" or
              .target_clip_uid == "23f996b204a2461d18c3cfea" or
              .target_clip_uid == "c54f78ddceb866055c4114a0") |
  {schema_version:"r2v.h3.omni_av_speaker_judge_manifest.1",
   clip_uid:.target_clip_uid, segment_id:.segment_id, human_label:null}' \
  "$NEW_ROOT/diarization/raw_segments.jsonl" > "$CONTROL_MANIFEST"

python tools/run_h3_omni_av_speaker_judge.py \
  --audio-production-root "$NEW_ROOT" \
  --case-manifest "$CONTROL_MANIFEST" \
  --output-root "$NEW_ROOT/omni_av_speaker_judge_pilot_v1" \
  --base-url http://127.0.0.1:8091/v1 \
  --served-model-name Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --checkpoint-id Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --media-mode file \
  --media-root /mnt/workspace
```

Outputs are atomically published as `manifest.jsonl`, `records.jsonl`,
`summary.json`, `raw/`, `media/`, and `review.html`. Existing output is preserved
unless `--overwrite` is explicit.
