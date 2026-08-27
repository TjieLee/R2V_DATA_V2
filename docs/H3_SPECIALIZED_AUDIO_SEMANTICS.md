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

## Current architecture

```text
                                      +-> Qwen3-VL global ----------+
canonical full audio -> Captioner ----+                              |
                    \                 +-> optional recaption rescue  |
                     \                                               +-> assemble
                      +-> Qwen3-Omni-Instruct Local -----------------+
```

### Captioner

`Qwen3-Omni-30B-A3B-Captioner` receives canonical full audio only. There is no
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
OR non_diegetic_music is null
```

A null `overall_soundscape` alone does not trigger recaption.

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
global_missing_semantics_recaption_once_v1
```

### Local semantics

`Qwen3-Omni-30B-A3B-Instruct` receives canonical full audio and reuses the
validated `h3_target_audio_semantics_v2` prompt/fallback path. Specialized Local
publishes only:

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

A failed Local record may therefore retain trusted speaker delivery while its
`temporal_audio_events` remain unavailable.

### Speaker/entity/time authority

DiariZen remains the exact speaker sample/time authority. Qwen3-ASR remains the
dialogue-text authority. The MLLM does not decide entity identity.

Assembly deterministically maps:

```text
speaker_cluster_id -> frozen entity_id
```

and reattaches `e1`, `e2`, etc. from inventory evidence. An unbound cluster
remains `entity_id=null`.

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

The recaption rescue changes the Global phase dependency: Global now needs both
Qwen3-VL and Captioner online.

```text
captioner         -> 8092 Captioner
global-semantics  -> 8000 Qwen3-VL + 8092 Captioner
local-semantics   -> 8091 Qwen3-Omni-Instruct
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

Global only, including possible recaption rescue:

```bash
"$R2V_PYTHON" tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase global-semantics \
  --captioner-max-inflight 1 \
  --global-vl-max-inflight 4
```

Local only:

```bash
"$R2V_PYTHON" tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase local-semantics \
  --local-instruct-max-inflight 1
```

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
- Local structured extraction reuses the validated Omni prompt/fallback path.
- Local harmless order/envelope differences are normalized programmatically.
- Local speaker semantics can survive an independent event-format failure when
  the complete speaker array is strictly recoverable.
- Assembly performs no model calls.

## Fingerprints

- Captioner fingerprint includes canonical audio hash, checkpoint, sampling, and
  Captioner policy.
- Global fingerprint includes primary Captioner dependency, Qwen3-VL prompt
  versions, fallback policy, and recaption-rescue policy.
- Local fingerprint includes canonical audio, speaker clusters, duration,
  checkpoint, prompt/fallback policy, field-salvage policy, and input modality.
- Assembly fingerprints all three upstream records.

A Global policy change does not invalidate Captioner or Local semantic identity.

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

Observed on the current 35-clip pilot after adding recaption rescue:

```text
recaption rescue attempted: 20 / 35
recaption rescue used:       4 / 35
overall_audio_description:  35 / 35 non-null
overall_soundscape:          35 / 35 non-null
non_diegetic_music:          17 / 35 non-null
```

These numbers are observations, not acceptance thresholds. `recaption_used`
means at least one missing Global field was filled. It does **not** mean that
music was necessarily recovered. A real reviewed case had an all-null primary
Global result; the second Captioner/Global pass recovered description and
soundscape but both Captioner samples still denied music, so
`non_diegetic_music` remained null.

This is the main remaining semantic limitation: one stochastic recaption reduces
single-sample omission risk but cannot guarantee recovery when both Captioner
samples make the same false-negative judgment. Human review remains required
before freezing the Global music policy for large-scale production.
