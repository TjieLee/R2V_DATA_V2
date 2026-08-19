# H3 Target Audio Caption MLLM Pilot V1/V2/V3

## Status

The first bounded 20-clip runtime completed 20/20 and its review bundle is
available. Those ASR-pilot clips are predominantly acoustically clean, so formal
QA was intentionally skipped: the set lacks useful positive background cases.
It remains clean/negative evidence for hallucination and abstention behavior,
but cannot measure recall on real ambience, music, or non-speech events.

The current stage is a model-free scout over all 75 frozen production targets.
This particular dataset contains fewer than 20 obvious background-rich clips,
so the positive pilot size is the dynamic, non-empty set selected by a human.
This is dataset-distribution-specific, not a global assumption; positives must
never be fabricated to meet a quota. Neither pilot publishes production
captions or modifies or feeds the deterministic final H3 renderer.

The producer reads these frozen sources without modifying them:

- `$AUDIO_RUN_ROOT/production/diarization`;
- `$AUDIO_RUN_ROOT/production/asr_v2`;
- `$AUDIO_RUN_ROOT/production/text_usability`;
- `$AUDIO_RUN_ROOT/production/audio`.

The ordered clip identities come directly from
`$AUDIO_RUN_ROOT/asr_v2_pilot20/inventory.json`. There is no second sampling
policy, parent quota, donor job, or production mode.

The background-audio scout instead enumerates every target in
`$AUDIO_RUN_ROOT/production/diarization/inventory.json` and writes only to:

```text
$AUDIO_RUN_ROOT/background_audio_scout
```

It computes temporal-union speech coverage plus PCM16 RMS/peak navigation
diagnostics with zero model calls. High non-speech energy is not a semantic
classification and never selects a clip automatically. Reviewers label clips
`background-rich`, `clean`, or `uncertain`; export contains every manually
labeled `background-rich` clip in deterministic source order and requires at
least one. Clean and uncertain clips are excluded.

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

Prompt V3 (`h3_dots3_target_audio_caption_v3`) deliberately removes the V2 audio
taxonomy. Dots3 returns only one nullable, short English
`background_audio_prompt` describing meaningful non-speech audio actually
audible and one nullable delivery/prosody string for each frozen speaker
cluster. It may freely mention music, ambience, effects, traffic, crowds,
footsteps, doors, machinery, nature, or other audible background content;
faint or partially masked accompaniment is included when audible. It does not
separately classify ambient scene, music presence/style/prominence, sound-event
lists, or acoustic style. Input transport, identity restrictions, and repair
behavior are unchanged.

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

- `r2v.h3.target_audio_caption.2`;
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

Build the zero-model-call scouting report before the positive pilot:

```bash
cd "$REPO"
"$R2V_PYTHON" tools/scout_h3_background_audio.py \
  --audio-run-root "$AUDIO_RUN_ROOT"

cd "$AUDIO_RUN_ROOT/background_audio_scout"
"$R2V_PYTHON" -m http.server 8765 --bind 127.0.0.1
```

After manual review, export `background_audio_pilot_selection.json`, then run the
same Target Audio Caption policy over exactly that selection:

```bash
"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-run-root "$AUDIO_RUN_ROOT" \
  --clip-selection-json /path/to/background_audio_pilot_selection.json
```

Prompt V3 output is written to
`$AUDIO_RUN_ROOT/target_audio_caption_background_pilot_v3`. It reuses the exact
manual selection for an A/B comparison while preserving both V1 at
`$AUDIO_RUN_ROOT/target_audio_caption_background_pilot` and V2 at
`$AUDIO_RUN_ROOT/target_audio_caption_background_pilot_v2`. Native-video
transport and the one-repair fail-closed policy remain unchanged.
