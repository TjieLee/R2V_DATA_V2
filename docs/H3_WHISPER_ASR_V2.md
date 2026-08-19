# H3 Whisper ASR V2 DiariZen-Segment Pilot

## Status

ASR V2 is a **CALIBRATION PILOT**. It compares the frozen ASR V1
LR-ASD-derived turns with formal production DiariZen segments while keeping the
Whisper-large-v3 checkpoint and decoder semantics unchanged. Pilot20 inference
is allowed. Full production inference is blocked pending human QA.

Formal DiariZen source:

```text
$AUDIO_RUN_ROOT/production/diarization
inventory fingerprint:
bc750320ab0488c122c18c8a70afb6c44c70dd5b9b5c001d791cfaf9ff4fb1d9
mapping policy: h3_diarizen_sparse_anchor_policy_v1
```

The source completed 75/75 targets with 179 raw segments and 81 clusters: 79
`candidate_mapped`, one `ambiguous`, one `unbound`, and zero `conflict`.
Production mapped 461.828 speaker-seconds, including 371.345 direct-anchor and
90.483 identity-propagated seconds. Forty segments used accepted canonical EOF
intersection; median positive overrun was 0.0305 seconds and maximum was 0.0805
seconds / 1288 samples. These 81 production clusters have not all received
human review; only the 22-cluster calibration pilot was reviewed 22/22 correct.

## Controlled A/B

Both versions use complete Whisper-large-v3, CT2 float16, exact PCM crop,
`task=transcribe`, `condition_on_previous_text=false`, `vad_filter=false`,
`word_timestamps=false`, and local files only.

```text
ASR V1: frozen LR-ASD-derived coalesced turns -> Whisper-large-v3
ASR V2: production DiariZen raw segments    -> Whisper-large-v3
```

V2 never calls the LR-ASD turn coalescer. It does not merge, split, pad, retime,
denoise, enhance, normalize, or VAD-resegment DiariZen segments. Every raw
segment is one independent Whisper job, including overlapping speakers. The
crop is exactly `[source_start_sample, source_end_sample)`, using the effective
canonical coordinates already published by DiariZen, never the backend-reported
end beyond EOF.

Pilot selection reuses only the exact ordered 20 clip IDs from
`$AUDIO_RUN_ROOT/asr_pilot20/inventory.json`. Segment jobs come exclusively
from formal production DiariZen artifacts. The dry-run derives the segment
count from those artifacts; there is no fixed 50-segment limit.

## Identity

Every V2 segment requires `speaker_cluster_id`. Entity identity is nullable:

- `candidate_mapped`: entity and occurrence IDs are present;
- `ambiguous`, `unbound`, or `conflict`: entity and occurrence IDs are null.

All four statuses are transcribed. Transcript availability is independent from
speaker identity, primary-voice quality, pair eligibility, and Visual assets.
Whisper receives only waveform and sample rate.

## Schemas And Roots

- `r2v.h3.asr_v2_segment.1`
- `r2v.h3.asr_v2_inventory.1`
- `r2v.h3.asr_v2_summary.1`
- `r2v.h3.asr_v2_human_qa.1`

```text
pilot:      $AUDIO_RUN_ROOT/asr_v2_pilot20
production: $AUDIO_RUN_ROOT/production/asr_v2
```

The pilot writes `inventory.json`, `segments.jsonl`, `summary.json`,
`review.html`, and `review_media/` atomically. The review includes exact crop
audio, identity status, decoder diagnostics, human QA controls, and temporally
overlapping ASR V1 text labeled `ASR V1 REFERENCE - NOT GROUND TRUTH`.

## Commands

```bash
"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --dry-run

"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --overwrite

"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode pilot20 \
  --regenerate-review

"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --mode production \
  --dry-run
```

A non-dry production command fails closed with
`production_blocked_pending_asr_v2_human_calibration`.

## Calibration Boundary

Decoder diagnostics remain observations, not correctness probabilities. No
language-probability, log-probability, no-speech, compression, duration, or
other transcript threshold is active. Human QA must precede any text-usability
policy. ASR V2 does not modify ASR V1, DiariZen, primary voice, embeddings,
pairs, Visual data, or final H3 rendering.
