# H3 ASR V2 Text Usability Calibration

## Status

This is a **MODEL-FREE CALIBRATION ANALYZER**. It does not freeze or apply a
text-usability policy. Formal raw ASR V2 production is complete and read-only;
the analyzer asks only whether a future renderer might trust a raw transcript
enough to display it inside `<d>...</d>`.

The three quality questions remain independent:

```text
transcript usability
!= speaker identity quality
!= voice-reference quality
```

Hiding text under a calibration candidate never invalidates the raw ASR record,
speaker cluster, entity mapping, primary voice, embeddings, pair assets, or
canonical audio.

## Frozen Inputs

The reviewed pilot is `$AUDIO_RUN_ROOT/asr_v2_pilot20`, with inventory
fingerprint
`e57635fa61541d4e1aaed6d49ccabc4bf85152d52432b0bb2e96c6f7a824ebb0`.
It contains 20 clips and 50 DiariZen segments: 48 transcribed, two backend
uncertain, and zero failed. Human QA is complete: 41 `CORRECT`, three `WRONG`,
six `UNCERTAIN`, and zero unlabeled.

The browser-exported QA JSON is supplied explicitly with `--qa-json`. The
analyzer validates schema, mode, fingerprint, exact frozen counts, and a strict
one-to-one `(target_clip_uid, segment_id, speaker_cluster_id)` join. It copies
the exact QA bytes and records their SHA-256 in the output.

Formal production under `$AUDIO_RUN_ROOT/production/asr_v2` is a read-only
shadow dataset. Its current 179 records are loaded dynamically; the analyzer
does not hardcode production segment count.

## Diagnostics And Rules

`raw_text_available` requires `status == transcribed` and non-empty text.
Backend-uncertain and failed records are never displayable. Candidate gates are
evaluated only on raw-text-available records.

The analyzer preserves duration, decoder segment count, and these decoder
diagnostics:

- `avg_log_probability`
- `no_speech_probability`
- `language_probability`
- `compression_ratio`

`language_probability` is a language-identification diagnostic. It is **not
transcript correctness probability** and is never described as such.

The baseline is `raw_text_available_only`. One-dimensional sweeps use observed
finite pilot values and midpoints that create distinct decisions; there is no
privileged or fixed `0.5` threshold. Missing values fail the condition
explicitly. A bounded pairwise grid evaluates conjunctions of two different
diagnostics, with at most nine observed decision thresholds per diagnostic.
There are no three-condition rules, weighted scores, or learned classifiers.

Every rule reports retained/rejected QA counts, correct retention, wrong
leakage, uncertain retention, explicit and conservative precision, retained
clip count, retained-WRONG clips, and an empirical normalized margin from the
nearest rejected WRONG example.

The report creates, but does not select:

- a zero-WRONG shortlist;
- an at-most-one-WRONG shortlist;
- a compact Pareto frontier minimizing wrong retention, uncertain retention,
  and complexity while maximizing correct retention.

Every candidate is labeled `CALIBRATION CANDIDATE ONLY`.

## Production Shadow

Shortlisted rules are applied to production as read-only coverage simulations.
The report includes raw-text-available, retained, and hidden counts, stratified
by cluster-binding status and identity scope. Identity is analysis context only
and is never a text-gate condition. Production has no human correctness labels,
so the report makes no production precision or accuracy claim.

## Artifacts And Command

The fixed, atomically published output root is:

```text
$AUDIO_RUN_ROOT/asr_v2_text_calibration/
├── inventory.json
├── human_qa.json
├── joined_segments.jsonl
├── sweep.json
├── summary.json
└── report.html
```

Run from the repository root:

```bash
"$R2V_PYTHON" tools/analyze_h3_asr_v2_text_usability.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --qa-json "$QA_JSON" \
  --overwrite
```

The command performs zero Whisper calls, zero DiariZen calls, zero GPU calls,
and zero network-model calls. It does not access model checkpoints. To inspect
the static report while retaining optional sibling-media links, serve the audio
run root:

```bash
cd "$AUDIO_RUN_ROOT"
python -m http.server 8768 --bind 127.0.0.1
```

Then open `asr_v2_text_calibration/report.html`.

## Decision Boundary

The published inventory and summary retain:

```text
text_usability_policy_validated = false
text_usability_gate_applied = false
transcript_confidence_threshold_used = false
```

Human review of the report is the next step. A later, separate decision may
freeze a simple rule if the evidence supports one. This analyzer does not alter
the renderer, generate `<d>` markup, filter sidecars, or mutate production ASR.
