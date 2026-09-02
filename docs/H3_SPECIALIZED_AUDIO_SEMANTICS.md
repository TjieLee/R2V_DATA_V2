# H3 Specialized Audio Semantics V1

This producer is an additive specialized-audio experiment beside the legacy
`r2v.h3.target_audio_caption.8` path. It does not rewrite legacy artifacts.

Fixed output root:

```text
$AUDIO_PRODUCTION_ROOT/audio_semantics_specialized_v1/
  inventory.json
  captioner/{records.jsonl,summary.json,raw/}
  global_semantics/{records.jsonl,summary.json,raw/}
  local_semantics/{records.jsonl,summary.json,raw/}
  assembled/{records.jsonl,summary.json,review.html,media/}
```

Every phase publishes atomically and keeps raw model responses, diagnostics,
request fingerprints, and upstream dependency fingerprints.

The clip-level admission universe is every validated row in Visual Production
`samples.jsonl`, represented by `audio/canonical_clips.jsonl`. Subject
references, primary voice, pairs, DiariZen speakers, and Qwen3-ASR transcripts
are optional enrichment and never remove a canonical clip from this producer.
The specialized inventory contract is
`r2v.h3.specialized_audio_semantics_inventory.1`; it reads the canonical Audio
manifest directly, does not read `pairs/in_pairs.jsonl` or ASR output, and treats
readable DiariZen rows as an optional subset overlay.

## Current architecture

```text
H3-native canonical audio -> Captioner -> Qwen3-VL Global
                         |    \       -> optional recaption rescue
                         |     \      -> direct-audio music rescue (8091)
                         +----------> Qwen3-Omni-Instruct Local hints (8091)
H3-native canonical audio ----------> Qwen3-Omni-Instruct Local truth (8091)
                                              |
                                              +-> assemble
```

### Captioner

`Qwen3-Omni-30B-A3B-Captioner` receives the canonical 32 kHz stereo H3-native
audio only. There is no
text prompt and no video. Its native whole-audio caption is preserved verbatim
as durable primary evidence.

Sampling is intentionally stochastic:

```text
temperature = 0.6
top_p       = 0.95
top_k       = 20
max_tokens  = 16384
```

### Global semantics

`Qwen3-VL-32B-Instruct` is text-only. It receives a Captioner raw caption and
publishes:

```text
overall_audio_description
overall_soundscape
non_diegetic_music
```

The current Global music policy is high-recall: clearly musical content with an
unspecified source may populate `non_diegetic_music`; explicit in-scene sources
such as radio, television, phone, loudspeaker, or a live/in-scene performance
are excluded. A generic hum, buzz, whine, drone, or tone alone is not upgraded
to music without additional musical evidence.

Global remains extraction-only relative to its Captioner input. Qwen3-VL must
not contradict a Captioner statement by inventing an unsupported audible fact.

### One-time Global recaption rescue

Because Captioner is stochastic, a single caption can miss or falsely negate an
audible fact. After the normal Global pass, one auxiliary recaption is allowed
when:

```text
primary Global status == failed
OR overall_audio_description is null
```

A null `overall_soundscape` or `non_diegetic_music` alone does not trigger
recaption.

Rescue flow:

```text
primary Captioner -> primary Global
                        |
                        +-- missing key semantics --> Captioner sample #2
                                                    -> Global sample #2
                                                    -> deterministic fill
```

The auxiliary Captioner result never replaces the canonical Captioner record.
For a ready primary Global record, rescue may fill only fields that are null:

```text
primary non-null field -> always keep primary
primary null + rescue non-null -> fill from rescue
primary null + rescue null -> remain null
```

No free-text concatenation or third model call is used. Rescue failure cannot
turn an already-ready primary Global record into a failed record. Raw artifacts
separate `primary_global` from `recaption_rescue` and retain both Captioner and
Qwen3-VL diagnostics.

Current policy version:

```text
global_failed_or_missing_description_recaption_once_v2
```

### Direct-audio music rescue

After primary Global and optional recaption finish, a ready Global record whose
`non_diegetic_music` is still null gets one music-only request to the existing
Qwen3-Omni-Instruct service on port 8091. The detector receives canonical full
audio only. It receives no Captioner text, transcript, speaker/entity evidence,
video, reference image, or Local response.

The detector may only fill `non_diegetic_music`; it cannot alter description,
soundscape, events, or speaker delivery. A schema-valid null is final and does
not trigger another semantic listen. Malformed JSON gets the existing one
format-repair attempt. Transport, media, or repair failure is auxiliary evidence
only and cannot turn a ready Global record into a failure.

This v2 detector is precision-oriented. It requires audible musical structure:
a clear melody or repeating melodic phrase, harmony/chord progression, rhythmic
musical pattern/beat, or clearly identifiable instrumental performance. An
ambient pad, electronic drone, hum, rumble, continuous tone, isolated tone, or
vague tonal texture is not sufficient by itself. Ambiguous music-versus-tonal
sound returns null, and a specific instrument is named only when its timbre is
actually clear. Upstream no-music text is not a veto; direct audio remains the
only evidence for this rescue.

```text
prompt: h3_non_diegetic_music_direct_audio_v2
policy: global_non_diegetic_music_direct_audio_once_v2
```

The canonical H3-facing field remains `non_diegetic_music`.

### Local semantics

`Qwen3-Omni-30B-A3B-Instruct` receives canonical H3-native audio plus the existing
canonical Captioner raw caption. The caption is untrusted acoustic search hints,
not truth: positive mentions only prompt active listening, hallucinated mentions
must not be copied, and negative statements cannot veto sound audible in the
audio. The attached audio remains the only factual source. Captioner evidence is
ignored completely for `speaker_delivery`, which uses only audio and frozen
speaker-cluster timing. Specialized Local publishes only:

```text
temporal_audio_events
speaker_delivery
```

Program-side normalization handles harmless formatting variation:

- valid temporal events are sorted deterministically;
- speaker delivery is reordered to inventory speaker-cluster order;
- exact `Assistant:` and complete JSON-fence wrappers are normalized;
- a failed full Local response may conservatively salvage a complete valid
  `speaker_delivery` array without salvaging malformed event times.
- obvious whole-string speech-only events such as `a person speaking`,
  `someone talking`, `speech`, or speech-plus-language/prosody templates are
  dropped; mixed descriptions such as `metallic clink while a person is
  speaking` remain intact. This is
  `specialized_local_drop_speech_only_events_v2`.

The event pass actively scans the full clip for reliably audible brief, quiet,
background, and speech-masked non-linguistic events. Ordinary dialogue is never
an event; laughter, coughs, gasps, sighs, physical sounds, and other genuinely
non-linguistic localized audio remain eligible.

A failed Local record may therefore retain trusted speaker delivery while its
`temporal_audio_events` remain unavailable.

Local prompt v2 asks for generic audible event descriptions rather than
uncertain source guesses and excludes background score/BGM/non-diegetic music
from temporal events. Localized in-scene playback or performance remains
eligible. When no speaker clusters are supplied, Local still runs the complete
event pass and must return `speaker_delivery=[]`; a schema-valid empty event and
speaker result is ready, not failed.

### Speaker/entity/time authority

DiariZen remains the exact speaker sample/time authority. Qwen3-ASR remains the
dialogue-text authority. The MLLM does not decide entity identity.

Assembly deterministically maps:

```text
speaker_cluster_id -> frozen entity_id
```

and reattaches `e1`, `e2`, etc. from inventory evidence. An unbound cluster
remains `entity_id=null`.

The assembled contract is `r2v.h3.specialized_audio_semantics.3`; its
`target_audio_binding_path` and hash are nullable for clips without subject
binding evidence. A no-speaker clip is still `complete` when Captioner, Global,
and Local all complete.

## Model-free canonical Audio backfill

Existing production roots can add missing no-subject canonical audio and publish the
same canonical manifest used by fresh Audio runs without rerunning LR-ASD,
Silero, DiariZen, ASR, or any model:

```bash
"$R2V_PYTHON" tools/backfill_h3_canonical_audio.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT"
```

Existing valid 32 kHz stereo FLAC files are reused. Missing canonical audio is
materialized directly from the processed canonical target video. The tool updates only
`audio/canonical_clips.jsonl` and `audio/canonical_clips_summary.json`; subject
bindings and all downstream stage directories remain untouched.
The CLI argument remains the whole `$AUDIO_PRODUCTION_ROOT`; `audio/` is selected
internally through the shared JEA production-path contract.

## Current runtime topology

The current specialized 35-clip pilot has been run with:

```text
GPU 0-3  Qwen3-VL-32B-Instruct       port 8000
GPU 4-5  Qwen3-Omni-Instruct TP2    port 8091
GPU 6-7  Qwen3-Omni-Captioner TP2   port 8092
```

The two Qwen3-Omni services are intentionally separate processes. Full paths,
TP2 deployment configuration, launch commands, health checks, and the historical
TP4 fallback are recorded in:

```text
docs/H3_QWEN3_OMNI_SERVING.md
```

## Runtime environment

```bash
export REPO=/mnt/workspace/litengjie/data/R2V_DATA_V2
export R2V_PYTHON="$REPO/.venv/bin/python"

export AUDIO_PRODUCTION_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/production/jea_motion_v1/e2e200-random-seed20260821-20260821-143232

export QWEN3_OMNI_CAPTIONER_BASE_URL='http://127.0.0.1:8092/v1'
export QWEN3_OMNI_CAPTIONER_API_KEY='EMPTY'
export QWEN3_OMNI_CAPTIONER_MODEL='Qwen/Qwen3-Omni-30B-A3B-Captioner'
export QWEN3_OMNI_CAPTIONER_CHECKPOINT_ID='/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Captioner'
export QWEN3_OMNI_CAPTIONER_MEDIA_MODE=file
export QWEN3_OMNI_CAPTIONER_MEDIA_ROOT=/mnt/workspace

export QWEN3_VL_AUDIO_SEMANTICS_BASE_URL='http://127.0.0.1:8000/v1'
export QWEN3_VL_AUDIO_SEMANTICS_API_KEY='EMPTY'
export QWEN3_VL_AUDIO_SEMANTICS_MODEL='/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct'
export QWEN3_VL_AUDIO_SEMANTICS_CHECKPOINT_ID='/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct'

export QWEN3_OMNI_LOCAL_BASE_URL='http://127.0.0.1:8091/v1'
export QWEN3_OMNI_LOCAL_API_KEY='EMPTY'
export QWEN3_OMNI_LOCAL_MODEL='Qwen/Qwen3-Omni-30B-A3B-Instruct'
export QWEN3_OMNI_LOCAL_CHECKPOINT_ID='/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Instruct'
export QWEN3_OMNI_LOCAL_MEDIA_MODE=file
export QWEN3_OMNI_LOCAL_MEDIA_ROOT=/mnt/workspace

export NO_PROXY='127.0.0.1,localhost'
export no_proxy='127.0.0.1,localhost'
```

## Service requirements by phase

The two Global rescues mean standalone Global may need all three services.

```text
captioner         -> 8092 Captioner
global-semantics  -> 8000 Qwen3-VL + 8092 Captioner + 8091 Omni-Instruct
local-semantics   -> 8091 Qwen3-Omni-Instruct + existing Captioner artifact
assemble          -> no model
pipeline          -> 8000 + 8091 + 8092
```

## Recommended phase commands

Captioner only:

```bash
"$R2V_PYTHON" tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase captioner \
  --captioner-max-inflight 1
```

Global only, including possible recaption and direct-audio music rescue:

```bash
"$R2V_PYTHON" tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase global-semantics \
  --captioner-max-inflight 1 \
  --global-vl-max-inflight 4 \
  --local-instruct-max-inflight 1
```

Local only:

```bash
"$R2V_PYTHON" tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase local-semantics \
  --local-instruct-max-inflight 1
```

Standalone Local reads `captioner/records.jsonl` for each clip's canonical hint
and dependency fingerprint. Port 8092 does not need to be online, but the
Captioner stage artifact must exist. A failed per-clip Captioner record still
allows Local to run audio-only with no hint.

Assemble only:

```bash
"$R2V_PYTHON" tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase assemble
```

Full pipeline:

```bash
"$R2V_PYTHON" tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase pipeline \
  --captioner-max-inflight 1 \
  --global-vl-max-inflight 4 \
  --local-instruct-max-inflight 1
```

For a Global-only prompt/policy change, prefer regenerating only Global and
Assembly. Do not unnecessarily resample the stochastic primary Captioner or
rerun Local.

## Retry and failure policy

- Captioner retries once only for missing/non-string/empty/whitespace output.
- Global structured extraction gets one repair and the existing semantic
  recheck behavior.
- Global may then perform at most one recaption rescue under the trigger above.
- Recaption rescue fills only missing Global fields and never overwrites a
  primary non-null field.
- A ready Global record with null music then gets at most one direct-audio
  music-only rescue; schema-valid null is not semantically retried.
- Full pipeline music rescue starts only after the Local phase has finished, so
  the two uses of port 8091 do not compete.
- Local structured extraction uses the specialized hint-aware prompt while
  retaining the validated Omni transport, structured repair, semantic fallback,
  and field-salvage lifecycle.
- Local harmless order/envelope differences are normalized programmatically.
- Local speaker semantics can survive an independent event-format failure when
  the complete speaker array is strictly recoverable.
- Assembly performs no model calls.

## Fingerprints

- Captioner fingerprint includes canonical audio hash, checkpoint, sampling, and
  Captioner policy.
- Global fingerprint includes primary Captioner dependency, Qwen3-VL prompt
  versions and fallback policy, recaption policy, plus the 8091 music model,
  checkpoint, media configuration, prompt, and rescue policy.
- Local fingerprint includes canonical audio, speaker clusters, duration,
  checkpoint, prompt/fallback, field-salvage, caption-hint and event-filter
  policies, input modality, the exact Captioner record fingerprint, and caption
  hash when available.
- Assembly fingerprints all three upstream records.

A Captioner artifact change invalidates Global, Local, and Assembly. A Global-only
configuration change does not invalidate Local; a Local-only change does not
invalidate Captioner or Global. In the streaming pipeline, each completed
Captioner record independently makes that clip eligible for both Global and
Local. Captioner failure blocks Global but still releases Local without a hint.
Direct-audio music rescue starts only after all Local work finishes, so the two
8091 workloads never compete.

## Review semantics

`assembled/review.html` uses canonical audio copied/hard-linked under
`assembled/media/`.

Interpret Local review labels as:

```text
[none]        = Local ready and confirmed empty
[unavailable] = Local failed; that field has no publishable evidence
```

A failed Local record may still show a valid salvaged Speaker delivery while
Temporal audio events are `[unavailable]`.

## Current pilot observations

Observed on the current 35-clip production QA before the v2 precision and Local
event-recall revision:

```text
recaption rescue attempted:  2 / 35
recaption rescue used:       2 / 35
music rescue attempted:     18 / 35
music rescue used:          13 / 35
overall_audio_description:  35 / 35 non-null
overall_soundscape:          35 / 35 non-null
non_diegetic_music:          30 / 35 non-null
```

These numbers are observations, not acceptance thresholds. `recaption_used`
means at least one missing Global field was filled. It does **not** mean that
music was necessarily recovered. A real reviewed case had an all-null primary
Global result; the second Captioner/Global pass recovered description and
soundscape but both Captioner samples still denied music, so
`non_diegetic_music` remained null.

Human review found false-positive music classifications and low Local event
recall. Those findings motivated the precision-oriented direct-audio music v2
policy and the audio-verified Captioner-hint Local event pass. The figures above
remain pre-change QA evidence, not post-change results or thresholds.
