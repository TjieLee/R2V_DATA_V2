# H3 Specialized Audio Semantics V1

This producer is an additive experiment beside the legacy
`r2v.h3.target_audio_caption.8` path. It does not replace or rewrite legacy
artifacts. Its fixed output root is:

```text
$AUDIO_PRODUCTION_ROOT/audio_semantics_specialized_v1/
  inventory.json
  captioner/{records.jsonl,summary.json,raw/}
  global_semantics/{records.jsonl,summary.json,raw/}
  local_semantics/{records.jsonl,summary.json,raw/}
  assembled/{records.jsonl,summary.json,review.html}
```

Every phase publishes atomically, validates directory ownership before an
explicit overwrite, and can reuse a compatible existing phase. Runtime
concurrency limits are operational settings and do not enter semantic request
fingerprints.

## Authority And Roles

```text
                         +-> Captioner -> raw caption -> Qwen3-VL global --+
canonical target audio -+                                               +-> assemble
                         +-> Qwen3-Omni-Instruct local -------------------+
```

- **Qwen3-Omni-30B-A3B-Captioner** receives only canonical full audio. It
  produces its native rich whole-audio caption with no injected task prompt.
  The exact raw caption is durable evidence.
- **Qwen3-VL-32B-Instruct** is text-only. It may select, remove, compress,
  normalize, and conservatively classify facts explicitly supported by the raw
  caption. It publishes only `overall_audio_description`,
  `overall_soundscape`, and `non_diegetic_music`. It discards speech, vocal,
  language, identity, and delivery content and must not resolve speculative
  sources.
- **Qwen3-Omni-30B-A3B-Instruct** receives canonical full audio by default and
  publishes only `temporal_audio_events` and `speaker_delivery`. An explicit
  `--include-video` option remains an A/B mode and changes provenance and the
  request fingerprint.
- **DiariZen** remains the exact speaker/sample-time authority.
- **Qwen3-ASR** remains the dialogue-text authority.
- **Primary and cross voice references** remain the voice identity/timbre
  conditioning assets.

Assembly is deterministic and performs no model call. It reattaches frozen
`entity_id` values to speaker delivery, keeps canonical path/hash evidence, and
publishes complete, partial, or failed records without hiding stage failures.
A Captioner failure blocks only the Global text extractor; Local processing can
still finish.

## Independent Runtime Configuration

The three roles use independent OpenAI-compatible endpoints and independent
thread-local clients:

```bash
export QWEN3_OMNI_CAPTIONER_BASE_URL=...
export QWEN3_OMNI_CAPTIONER_API_KEY=EMPTY
export QWEN3_OMNI_CAPTIONER_MODEL=Qwen/Qwen3-Omni-30B-A3B-Captioner
export QWEN3_OMNI_CAPTIONER_CHECKPOINT_ID=/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Captioner
export QWEN3_OMNI_CAPTIONER_MEDIA_MODE=file
export QWEN3_OMNI_CAPTIONER_MEDIA_ROOT=/mnt/workspace

export QWEN3_VL_AUDIO_SEMANTICS_BASE_URL=http://127.0.0.1:8000/v1
export QWEN3_VL_AUDIO_SEMANTICS_API_KEY=EMPTY
export QWEN3_VL_AUDIO_SEMANTICS_MODEL=/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
export QWEN3_VL_AUDIO_SEMANTICS_CHECKPOINT_ID=/mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct

export QWEN3_OMNI_LOCAL_BASE_URL=...
export QWEN3_OMNI_LOCAL_API_KEY=EMPTY
export QWEN3_OMNI_LOCAL_MODEL=/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Instruct
export QWEN3_OMNI_LOCAL_CHECKPOINT_ID=/mnt/workspace/public/pretrained/Qwen/Qwen3-Omni-30B-A3B-Instruct
export QWEN3_OMNI_LOCAL_MEDIA_MODE=file
export QWEN3_OMNI_LOCAL_MEDIA_ROOT=/mnt/workspace
```

Captioner and Local ports are deliberately not defaulted because no specialized
serving ports have been validated. The existing Qwen3-VL endpoint at port 8000
is the only validated default recorded here.

## Recommended Phased Operation

Models do not need to be online simultaneously. Run only the phase whose
service is available, switch services if needed, and reuse the durable phase
outputs:

```bash
python tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase captioner \
  --captioner-max-inflight 1

python tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase global-semantics \
  --global-vl-max-inflight 4

python tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase local-semantics \
  --local-instruct-max-inflight 1

python tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase assemble
```

Use `--overwrite` only for the phase being regenerated. It never deletes a
different phase.

When all three endpoints are intentionally online, the explicit pipeline mode
uses a bounded streaming DAG. Captioner and Local work overlap, each completed
caption becomes immediately eligible for Global extraction, and final records
are still published in inventory order:

```bash
python tools/run_h3_specialized_audio_semantics.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --phase pipeline \
  --captioner-max-inflight 1 \
  --global-vl-max-inflight 4 \
  --local-instruct-max-inflight 1
```

`--dry-run` builds and validates the inventory without constructing a model
backend or writing phase artifacts.

## Retry And Fingerprint Policy

- Captioner retries exactly once only for missing, non-string, empty, or
  whitespace output. Ordinary request/media failures fail closed.
- Global and Local structured output get exactly one repair. A valid all-null
  response or whitespace response gets one semantic recheck. A confirmed
  all-null fallback is valid evidence, not a stage failure. Responses are never
  field-merged.
- Captioner fingerprints include canonical audio hash and backend/policy
  configuration.
- Global fingerprints include exact raw-caption hash and extractor model,
  prompt, and fallback policy.
- Local fingerprints include canonical audio, speaker-cluster input, duration,
  model, prompt/fallback policy, and input modality. Video hash is included only
  in explicit video mode.
- Assembly fingerprints all three upstream records.

Raw model responses and completion diagnostics remain in each phase's `raw/`
directory. `assembled/review.html` exposes raw Captioner evidence, all semantic
layers, per-stage provenance/status, and a specialized localStorage namespace.

## Deployment Status

The current logical target keeps the validated Qwen3-VL service on GPU 0-3 at
TP1 x DP4 and port 8000. The remaining GPUs may be evaluated as TP2
Qwen3-Omni-Instruct plus TP2 Captioner. That split is **not validated** and must
not be frozen until real memory and throughput smoke tests pass. Historical
Qwen3-Omni-Instruct TP4 serving remains a validated legacy topology; this
specialized producer does not require it or assume any GPU IDs, tensor-parallel
size, GPU count, or port.
