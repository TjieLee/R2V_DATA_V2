# H3 Target Audio Caption

## Current JEA H3 audio semantics

The current JEA production sidecar reads only:

```text
$AUDIO_PRODUCTION_ROOT/pairs/in_pairs.jsonl
$AUDIO_PRODUCTION_ROOT/diarization/readable_segments.jsonl
$AUDIO_PRODUCTION_ROOT/asr/segments.jsonl
```

`pairs/in_pairs.jsonl` owns target video/full-audio identity. Readable DiariZen
rows own exact speech intervals, `speaker_cluster_id`, and nullable entity
binding. Qwen3-ASR rows are reconciled exactly against those DiariZen identities,
sample ranges, time ranges, and bindings, but transcript text is never placed in
the caption inventory or model request. Whisper ASR-V2 and TextUsabilityPolicy are
not inputs to this path.

The readable LR-ASD audio and canonical final full audio may use different paths
and encodings. Inventory validation retains a strict whole-clip timeline sanity
gate with a 0.10-second tolerance for LR-ASD 25fps, ffmpeg, and container duration
quantization; readable segment sample ranges and both audio timeline bounds remain
fail-closed.

Both backends use primary prompt `h3_target_audio_semantics_v1` and the same
strict reusable semantic response:

```json
{
  "overall_soundscape": "global ambience and recurring sound or null",
  "non_diegetic_music": "audience-only score or null",
  "temporal_audio_events": [
    {
      "start_time": 5.1,
      "end_time": 5.8,
      "description": "brief laughter immediately after the speech"
    }
  ],
  "speaker_delivery": [
    {
      "speaker_cluster_id": "speaker_0",
      "delivery_style": "concise prosody or null"
    }
  ]
}
```

The canonical ontology separates global recurring ambience, non-diegetic score,
localized audible events, and speaker-specific delivery. `overall_soundscape`
contains ambience, room tone, environmental noise, and recurring physical or
non-verbal sound. `non_diegetic_music` contains audience-only score or BGM.
`temporal_audio_events` contains concise English descriptions and approximate
clip-relative ranges for discrete audible events, including diegetic music and
non-verbal human events. Overlap is allowed, but events must be chronological and
remain within the canonical clip duration plus the existing timeline tolerance.
These model-estimated event times are semantic evidence, not authoritative sample
boundaries.

Qwen3-ASR remains the owner of linguistic speech text. DiariZen remains the owner
of exact speaker and sample timing. Primary and cross voice-reference assets are
unchanged. This stage assigns no training task, copy relationship, sampling
policy, or rendering behavior; later training code may choose how to consume the
reusable facts and assets. Visual evidence may only disambiguate an already
audible sound and can never establish that a sound exists.

Code requires every supplied cluster exactly once and in order, then reattaches
the frozen nullable `entity_id`. The model receives neither transcript nor entity
identity. Dots3 receives the native target video with embedded audio. The
Qwen3-Omni Instruct backend receives both the whole target video and canonical
full audio, and requests text output only. Qwen3-Omni-Captioner is not used because
it does not accept the task's text prompt. Both backends fail closed after at most
one primary structured-output repair.

### Qwen semantic fallback

Human QA found occasional complete abstention, and real production observed
Qwen3-Omni generating only whitespace on an initial request or structured repair.
Qwen3-Omni therefore performs exactly one semantic reinspection using
`h3_target_audio_semantics_v1_recheck`. Primary and fallback return the same new
schema. Fallback triggers only when all four semantic layers are empty/null or
when primary processing ends with `qwen3_omni_vllm_empty_response`. Partial nulls
are valid. Connection, HTTP, timeout, model, local-media, and other ordinary
infrastructure failures do not trigger semantic fallback.

If fallback recovers any semantic value, its complete response becomes final;
fields are never merged. If fallback is also all-null, the result is ready and
there is no third attempt. Failed fallback preserves a valid all-null primary;
after whitespace-only primary failure it remains failed because no valid primary
exists. Dots3 never uses semantic fallback.

Primary structured-output repair and Qwen semantic fallback are separate
mechanisms. Summary counters report primary initial/repair calls separately from
fallback triggers by reason, initial calls, repair calls, recoveries, confirmed
all-null results, and fallback failures. Each record publishes the explicit
trigger reason and semantic source. Raw diagnostics retain both passes, ordered
completion metadata (`finish_reason`, optional usage, and whitespace character
counts), and the validation issues that caused a repair even if that repair later
fails. Fallback policy versioning participates in Qwen request fingerprints;
runtime concurrency does not.

The new contracts are:

- `r2v.h3.target_audio_caption.6`;
- `r2v.h3.target_audio_caption_inventory.3`;
- `r2v.h3.target_audio_caption_summary.6`;
- `r2v.h3.target_audio_caption_human_qa.3`.

Run model-free inventory validation first:

```bash
"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --backend dots3 \
  --dry-run

"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --backend qwen3-omni \
  --dry-run
```

For Dots3 configure only `DOTS3_BASE_URL`, `DOTS3_API_KEY`, `DOTS3_MODEL`,
`DOTS3_CHECKPOINT_ID`, `DOTS3_MEDIA_MODE`, `DOTS3_MEDIA_ROOT`, and optionally
`DOTS3_MEDIA_BASE_URL`. For Qwen3-Omni use the corresponding `QWEN3_OMNI_*`
variables. Variables for the unselected backend are not required. The Qwen3-Omni
served-model/checkpoint default is `Qwen/Qwen3-Omni-30B-A3B-Instruct`; deployment
may override it explicitly without changing the shared semantic schema.

Run each A/B side independently:

```bash
"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --backend dots3 \
  --max-concurrency 4

"$R2V_PYTHON" tools/run_h3_target_audio_caption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --backend qwen3-omni \
  --max-concurrency 4
```

`--max-concurrency` controls how many independent target clips the producer
sends to the serving backend at once. It is not a GPU count or a vLLM tensor
parallel setting, and it is deliberately excluded from semantic request
fingerprints. Each clip's primary request, optional structured repair, and
optional Qwen semantic fallback remain sequential. The default value `1` is
the compatibility and debugging mode. For the current four-GPU evaluation
deployment, start with `4`; benchmark `1`, `2`, `4`, and `8` against the actual
serving topology before freezing an operational value. No linear scaling is
assumed.

Outputs are atomically published and cannot overwrite one another:

```text
$AUDIO_PRODUCTION_ROOT/audio_caption/dots3/
$AUDIO_PRODUCTION_ROOT/audio_caption/qwen3_omni/
```

Each contains `inventory.json`, `records.jsonl`, `summary.json`, `raw/`,
`media/`, and `review.html`. Review renders soundscape, non-diegetic music,
temporal events, and speaker delivery separately. Its localStorage namespace is
derived from schema, prompt, backend family, backend configuration fingerprint,
and inventory fingerprint, so stale labels cannot cross semantic configurations.
The static review exports deterministic QA with backend/model/checkpoint/
configuration provenance plus `CORRECT`, `WRONG`, or `UNCERTAIN` and the approved
field-specific failure flags. API keys are never persisted.

## Historical ASR-V2 pilot

Everything below this point is retained for old V1/V2/V3 pilot provenance. It
does not define the current JEA CLI, inputs, schema versions, or output roots.

### Status

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
