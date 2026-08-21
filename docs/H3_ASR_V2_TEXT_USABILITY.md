# H3 ASR V2 Text Usability

## Status

`TextUsabilityPolicy V1` is **COMPLETE / FROZEN**. It is a model-free,
deterministic display-eligibility policy over immutable ASR V2 production
records. It does not alter raw ASR, segment timing, speaker/entity identity,
primary voice, embeddings, or pairs, and it does not render `<d>` markup.

The three quality questions remain independent:

```text
transcript usability
!= speaker/entity identity usability
!= voice-reference usability
```

Hidden text remains available as raw ASR evidence in the source production
record. Hiding it does not mean the voice, speaker, entity, or segment is bad.

## Frozen Policy

Policy version:

```text
h3_asr_v2_text_usability_policy_v1
```

A transcript is trusted for later display if and only if:

```text
status == transcribed
AND raw text is non-empty
AND language_probability >= 0.65
```

Otherwise the sidecar publishes `text_status=hidden` and
`trusted_text=null`. The threshold is exactly `0.65`, an intentionally rounded,
human-approved eligibility threshold selected after calibration. It is not the
observed value `0.680419921875`, `0.516479492188`, an automatically optimized
threshold, or a generic `0.5` probability cutoff.

`language_probability` is a language-identification diagnostic. It is **not
transcript correctness probability**. No log-probability, no-speech,
compression-ratio, duration, weighted, multi-condition, or learned gate is part
of V1.

## Calibration Provenance

The frozen calibration source is `$AUDIO_RUN_ROOT/asr_v2_pilot20`, inventory
fingerprint
`e57635fa61541d4e1aaed6d49ccabc4bf85152d52432b0bb2e96c6f7a824ebb0`.
It contains 50 DiariZen segments with complete human QA: 41 `CORRECT`, three
`WRONG`, six `UNCERTAIN`, and zero unlabeled.

The earlier model-free analyzer and its evidence remain unchanged at:

```text
$AUDIO_RUN_ROOT/asr_v2_text_calibration
```

That analyzer's sweeps remain calibration evidence, not production execution.
The V1 production implementation is a separate explicit predicate and does not
depend on sweep ranking or a calibration output directory.

## Production Source And Output

The source is read-only:

```text
$AUDIO_RUN_ROOT/production/asr_v2
inventory fingerprint:
53550fd6206f90368023caaf5629eab3b9cc6e7a5ca9f9b8081d6e5d49c173de
```

The current formal source contains 75 clips and 179 segments: 176 transcribed,
three backend-uncertain, and zero failed. Runtime enumeration is dynamic; 179
is provenance, not an execution cap.

The fixed, atomically published output is:

```text
$AUDIO_RUN_ROOT/production/text_usability/
├── inventory.json
├── segments.jsonl
└── summary.json
```

Schemas:

- `r2v.h3.text_usability_segment.1`
- `r2v.h3.text_usability_inventory.1`
- `r2v.h3.text_usability_summary.1`

Every source segment receives one sidecar row in authoritative inventory order.
Trusted text is copied character-for-character from raw ASR. Hidden rows retain
source identity, request provenance, language probability, and raw-text hash,
but do not duplicate hidden transcript content.

The only hidden reason codes are:

- `raw_text_unavailable`
- `language_probability_unavailable`
- `language_probability_below_threshold`

Summary identity breakdowns are diagnostic only. `candidate_mapped`,
`ambiguous`, `unbound`, and `conflict`, plus direct/propagated/unresolved scope,
never participate in the text predicate.

## Commands

Dry-run requires no model environment:

```bash
"$R2V_PYTHON" tools/run_h3_text_usability.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --dry-run
```

Publish once:

```bash
"$R2V_PYTHON" tools/run_h3_text_usability.py \
  --audio-run-root "$AUDIO_RUN_ROOT"
```

Use `--overwrite` only for an intentional deterministic replacement of this
derived sidecar. The command verifies source inventory identity and source-file
hash stability before atomic publication. It performs zero Whisper calls, zero
DiariZen calls, zero GPU calls, and zero network-model calls.

## Decision Boundary

This stage establishes nullable `trusted_text` only. A future renderer may emit
`<d>trusted_text</d>` when non-null and omit dialogue text otherwise. That
renderer, entity-to-subject binding, unresolved-only MLLM handling, and optional
enhancement experiments are not implemented here.
