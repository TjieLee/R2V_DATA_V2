# Deterministic H3 Final Renderer V1

## Status

> **Legacy path:** This document describes the older Whisper ASR V2,
> text-usability, and copied `assets/` renderer. It is not the active JEA Qwen3
> H3 production path. The current renderer contract and commands are documented
> in `docs/H3_QWEN3_ASR.md`; in particular, current Visual references publish a
> directly readable `image_artifact_path` and do not copy Visual assets into H3.

## Active JEA Canonical Renderer

The active JEA renderer is rooted in the complete canonical Visual inventory,
not in `pairs/in_pairs.jsonl`. It requires exact canonical Visual/Audio and
DiariZen target coverage, a model-free `binding_audit_v1`, and Qwen3-ASR rows
for every actual DiariZen segment. A DiariZen `empty` result is a valid
no-speech target; a DiariZen `failed` result blocks final publication.

The current contracts are `r2v.h3.final_sample.5` and
`r2v.h3.final_summary.6`. Every canonical clip produces exactly one
`pair_type="canonical"` base sample with no voice reference. Valid in-pairs and
cross-pairs add optional target-voice and donor-voice variants without deciding
whether the canonical target survives. Pair artifacts are therefore an
optional strict subset of the canonical target inventory.

Final `.5` publishes one canonical audio representation.
`target_full_audio_path` is the directly readable 32 kHz stereo lossless FLAC
decoded from the original target video stream, and
`subject_voices[].voice_reference_path` is a 32 kHz stereo crop from the target
or donor canonical audio. DiariZen, Qwen3-ASR, and ECAPA derive deterministic
16 kHz mono runtime views without publishing another canonical audio tree. All
persisted source sample indices use the canonical 32 kHz domain.

`full_clip_audio_semantics` is a backend-neutral projection of a validated
specialized assembled record. It carries final descriptions, temporal
non-speech events, and speaker delivery, but never the upstream raw caption.
If no semantics root is supplied, canonical samples still publish with a null
semantics field and explicit missing-coverage counts in `summary.json`.

Active output is fixed at `$AUDIO_PRODUCTION_ROOT/h3/`; per-clip files include
`canonical.json` and, when available, `in_pair.json` and
`cross_pair_1.json`. Source media and Visual references remain directly
readable provenance paths and are not copied.

The deterministic final renderer is implemented as a data-assembly stage. It
does not run Whisper, DiariZen, an MLLM, or any other model. Its inputs remain
read-only and its fixed output is:

```text
$AUDIO_RUN_ROOT/production/h3/
  inventory.json
  samples.jsonl
  summary.json
  failures.jsonl
  assets/
    videos/
    full_audio/
    pictures/
    voices/
```

Publication uses a temporary sibling directory and an atomic directory rename.
`--overwrite` atomically replaces only `production/h3`; it never modifies pair,
DiariZen, ASR V2, text-usability, Visual, or source media artifacts.

## Frozen Inputs

The renderer requires the complete frozen production artifacts:

- `production/pairs/in_pairs.jsonl` and `cross_pairs.jsonl`;
- `production/diarization/`;
- `production/asr_v2/`;
- `production/text_usability/`;
- each target clip's accepted Visual V3 `clip.json` and canonical
  `instruction.r2v_instruction`.

Every source fingerprint and file hash is checked before publication. The
renderer uses `trusted_text` from `h3_asr_v2_text_usability_policy_v1` as the
authoritative display text and never recalculates the `0.65` policy gate.

## Final Contracts

The old draft `r2v.h3.sample.1` contract is unchanged. Final output uses:

- `r2v.h3.final_sample.1`;
- `r2v.h3.final_speech_segment.1`;
- `r2v.h3.final_inventory.1`;
- `r2v.h3.final_summary.1`.

Pair order determines `subject_1`, `subject_2`, and later subjects. An in-pair
uses the target picture and target primary voice. A cross-pair changes only the
voice asset to the donor voice; video, full audio, picture, speech timeline,
transcript, and Visual instruction remain target-owned.

Mapped speech is subject-bound only when its target entity occurs in the pair.
Mapped additional speakers and unresolved speakers retain structured evidence
with a null `subject_id`. Trusted subject-bound text is rendered exactly as:

```text
<Subject N> says <d>TRUSTED_TEXT</d>
```

Trusted text without a subject remains in `speech_segments` with
`trusted_text_unrendered_no_subject` and is not assigned an invented subject.
Hidden text is not rendered. Overlapping segments remain separate and are
ordered by start, end, segment ID, and speaker-cluster ID.

Detected `zh`, `en`, `de`, or other ASR language codes are metadata only.
`language_conditioning_applied` is always false; the renderer adds no language
prompt lines or language tokens.

## Commands

Read-only plan:

```bash
"$R2V_PYTHON" tools/run_h3_final_renderer.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --dry-run
```

Atomic publication:

```bash
"$R2V_PYTHON" tools/run_h3_final_renderer.py \
  --audio-run-root "$AUDIO_RUN_ROOT"
```

Intentional replacement of only the final derived output:

```bash
"$R2V_PYTHON" tools/run_h3_final_renderer.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --overwrite
```

Inspect without any model environment:

```bash
python -m json.tool "$AUDIO_RUN_ROOT/production/h3/summary.json"
wc -l \
  "$AUDIO_RUN_ROOT/production/h3/samples.jsonl" \
  "$AUDIO_RUN_ROOT/production/h3/failures.jsonl"
```

There is no sample limit, parent quota, or in/cross mixing ratio in this stage.
Unresolved-only MLLM enrichment and speech-enhancement experiments remain
optional future work and are not baseline rendering dependencies.
