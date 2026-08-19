# H3 Target Audio Caption MLLM Pilot V1

## Status

This is a bounded, human-reviewed 20-clip pilot. It does not publish production
captions and does not modify or feed the deterministic final H3 renderer.

The producer reads these frozen sources without modifying them:

- `$AUDIO_RUN_ROOT/production/diarization`;
- `$AUDIO_RUN_ROOT/production/asr_v2`;
- `$AUDIO_RUN_ROOT/production/text_usability`;
- `$AUDIO_RUN_ROOT/production/audio`.

The ordered clip identities come directly from
`$AUDIO_RUN_ROOT/asr_v2_pilot20/inventory.json`. There is no second sampling
policy, parent quota, donor job, or production mode.

## Ownership

- DiariZen owns `speaker_cluster_id` and segment timing.
- Deterministic DiariZen/Visual binding owns nullable `entity_id`.
- Whisper owns raw transcript and detected language.
- TextUsabilityPolicy V1 owns `trusted_text`.
- Dots3 supplies only audible ambience, music, non-speech events, acoustic
  style, and speaker delivery/prosody.

Transcript, trusted text, language, and entity identity are never supplied to
Dots3. The model returns exact cluster IDs only; code attaches the frozen
nullable `entity_id` after structured-output validation.

## Input Contract

The request contains text plus the original target video as one `video_url`.
Dots3Note consumes the video's embedded native audio. The canonical extracted
full-audio file remains path/hash evidence and is verified before inference and
before publication, but it is not sent as an `audio_url` item. No denoising,
enhancement, source separation, donor media, or target primary voice is used.

The recorded modality is:

```text
native_target_video_with_embedded_audio
```

## Model Prompt

The exact system prompt is defined as `SYSTEM_PROMPT` in
`r2v_data_v2/h3/target_audio_caption.py`. Its hard rules are:

- analyze audible evidence only;
- never infer sounds from visual content;
- never transcribe, quote, paraphrase, correct, or summarize dialogue;
- never infer speaker/entity identity, gender, age, nationality, or timbre;
- return null or empty lists when evidence is unclear;
- emit every supplied speaker cluster exactly once and no `entity_id`;
- return one compact schema-valid JSON object.

Malformed JSON and cluster-set violations receive exactly one constrained
repair. A second invalid response fails that clip closed while neighboring
clips continue.

## Artifacts

The fixed output root is:

```text
$AUDIO_RUN_ROOT/target_audio_caption_pilot20
```

It contains:

```text
inventory.json
records.jsonl
summary.json
report.html
raw/<clip_uid>.json
media/<clip_uid>.<audio-extension>
```

Schemas are:

- `r2v.h3.target_audio_caption.1`;
- `r2v.h3.target_audio_caption_inventory.1`;
- `r2v.h3.target_audio_caption_summary.1`;
- `r2v.h3.target_audio_caption_human_qa.1`.

`audio_prompt_draft` is rendered deterministically from the structured fields.
It is review-only and is not inserted into final H3 samples.

## Human QA

`report.html` plays the canonical target audio and displays the structured
result, deterministic draft, and authoritative cluster time ranges. Reviewers
select `CORRECT`, `WRONG`, or `UNCERTAIN` and may flag hallucinated music,
hallucinated sound events, wrong ambience, wrong delivery style, dialogue
leakage, or another issue. The browser exports
`target_audio_caption_pilot20_human_qa.json`; labels are QA evidence only.

## Server Pilot

With the already-running validated Dots3 vLLM service and environment variables
from `docs/SERVER_AUDIO_PILOT_RUNBOOK.md`:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --dry-run

"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-run-root "$AUDIO_RUN_ROOT"
```

Use `--overwrite` only for an intentional pilot rebuild. The current stage has
no 75-clip production command. Human QA, production approval, final-renderer
integration, and a rebuilt final H3 dataset remain pending.
