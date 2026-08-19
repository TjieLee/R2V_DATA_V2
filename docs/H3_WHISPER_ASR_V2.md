# H3 Whisper ASR V2 Production Raw-ASR Baseline

## Status

The DiariZen-segment ASR V2 calibration is **COMPLETE / FROZEN** and accepted
for formal raw-ASR production. Production emits transcript observations,
decoder diagnostics, and segment/speaker/entity provenance. It does not decide
whether text is usable in final H3 rendering and does not make raw Whisper text
ground truth.

Formal DiariZen source:

```text
$AUDIO_RUN_ROOT/production/diarization
inventory fingerprint:
bc750320ab0488c122c18c8a70afb6c44c70dd5b9b5c001d791cfaf9ff4fb1d9
mapping policy: h3_diarizen_sparse_anchor_policy_v1
```

The source contains 75 targets, 179 raw segments, and 81 clusters: 79
`candidate_mapped`, one `ambiguous`, one `unbound`, and zero `conflict`. These
81 production clusters were not all human-reviewed; only the 22-cluster
DiariZen mapping calibration pilot was reviewed 22/22 correct.

## Accepted Calibration

ASR V2 used the exact ordered 20 clips from ASR V1 and changed only speech
segmentation. Both runs used complete Whisper-large-v3, CT2 float16, exact PCM
crop, `task=transcribe`, `condition_on_previous_text=false`,
`vad_filter=false`, `word_timestamps=false`, and local files only.

| Baseline | Units | Median duration | Human QA |
| --- | ---: | ---: | --- |
| ASR V1 LR-ASD-derived turns | 82 | about 1.02 s | 59 CORRECT / 15 WRONG / 8 UNCERTAIN |
| ASR V2 DiariZen segments | 50 | 2.14575 s | 41 CORRECT / 3 WRONG / 6 UNCERTAIN |

ASR V2 runtime produced 48 transcribed, two backend-uncertain, and zero failed
records. Backend `uncertain_count=2` and human-QA `UNCERTAIN=6` are different
concepts. All 50 V2 units were labeled; the accepted calibration inventory
fingerprint is:

```text
e57635fa61541d4e1aaed6d49ccabc4bf85152d52432b0bb2e96c6f7a824ebb0
```

The accepted complete Whisper-large-v3 checkpoint fingerprint is
`10ea6fb8ae7cdd1fa26495deeb1f32e79c1fc882c19f80ae711bf5f0dd671db3`.
Production validates the runtime against the frozen V1 model/decode provenance;
only the physical CUDA device index may differ.

The evaluation units differ, so this is not a paired per-turn accuracy delta.
The accepted conclusion is that DiariZen segmentation substantially reduced
explicitly wrong reviewed units in the controlled same-clip/same-model A/B and
is suitable as the production raw-ASR segmentation baseline.

## Frozen Semantics

Each formal DiariZen raw/effective segment is exactly one Whisper job. V2 does
not merge, split, pad, retime, denoise, enhance, normalize, add neighboring
context, or run VAD resegmentation. The crop is exactly
`[source_start_sample, source_end_sample)` and uses accepted canonical EOF
reconciliation, never the backend-reported overrun.

Every segment requires `speaker_cluster_id`. Entity and occurrence IDs remain
nullable. `candidate_mapped`, `ambiguous`, `unbound`, and `conflict` segments
all enter Whisper; unresolved identity never suppresses speech. Whisper receives
only waveform and sample rate.

There is no transcript confidence or text-usability gate. Language probability,
log probability, no-speech probability, compression ratio, and duration remain
diagnostics. Transcript quality remains independent from speaker identity,
primary voice, embeddings, and pair eligibility.

## Schemas And Roots

Frozen pilot artifacts remain readable under:

- `r2v.h3.asr_v2_inventory.1`
- `r2v.h3.asr_v2_summary.1`

Formal production adds calibration provenance under:

- `r2v.h3.asr_v2_inventory.2`
- `r2v.h3.asr_v2_summary.2`

The segment and QA contracts remain unchanged:

- `r2v.h3.asr_v2_segment.1`
- `r2v.h3.asr_v2_human_qa.1`

```text
frozen pilot: $AUDIO_RUN_ROOT/asr_v2_pilot20
production:   $AUDIO_RUN_ROOT/production/asr_v2
```

Production dynamically enumerates every formal DiariZen segment. No target,
parent, segment, donor, or cross-pair quota exists. Outputs are published
atomically. Review regeneration remains model-free and preserves inference JSON.

## Commands

```bash
"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" --mode production --dry-run

"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" --mode production

"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" --mode production --overwrite

"$ASR_ENV/bin/python" tools/run_h3_asr_v2_transcription.py \
  --audio-run-root "$AUDIO_RUN_ROOT" --mode production --regenerate-review
```

Future work remains transcript-usability calibration, unresolved-only MLLM,
optional enhancement experiments, entity-to-subject mapping, and final H3
rendering. None is part of the production raw-ASR stage.
