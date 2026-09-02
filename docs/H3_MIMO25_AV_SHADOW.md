# H3 MiMo-V2.5 AV Shadow

This experimental path is additive and read-only with respect to the current JEA
production stages. It writes only `mimo25_av_reconcile_v2/` and
`mimo25_h3_shadow_v2/` under the Audio production root.

## Authority contract

- DiariZen owns exact speech segment times and sample boundaries.
- Qwen3-ASR owns exact transcript text and language.
- frozen Visual V3 references own entity inventory, order, and image content.
- LR-ASD, source clusters, and current entity bindings are proposals.
- MiMo-V2.5 reconciles clip-local speaker groups and visible entities, describes
  AV-grounded non-speech audio, and writes an H3 visual/temporal draft.
- the deterministic materializer owns `Sx`, Subject and Audio references, exact
  dialogue, and final H3 formatting.

MiMo never splits or deletes a DiariZen segment. Multiple vocal events can mark
a segment as requiring acoustic refinement, but the authoritative segment and
Qwen3-ASR text remain present. The first inventory scope is explicitly
`current_diarization_asr_target_inventory`; `canonical_wide_coverage=false`, so
this work does not resolve the separate canonical-wide inventory gap.

Prompt, policy, annotation schema, and materializer versions are:

- `h3_mimo25_unified_av_reconcile_v2`
- `h3_mimo25_av_authority_contract_v2`
- `r2v.h3.mimo25_av_annotation.2`
- `h3_mimo25_materializer_v2`

The OpenAI-compatible client defaults to model `mimo-v2.5`, video FPS 4,
`media_resolution=default`, disabled thinking, JSON-object output,
temperature 0.2, and 16384 completion tokens. Base64 is the pilot default.
Payloads and API keys are never persisted. If reported embedded-audio tokens
are exactly zero, one request may retry with canonical full audio; unavailable
usage details do not trigger that fallback.

The exact request contract keeps `fps` and `media_resolution` beside the
`video_url` object, not inside it:

```json
{
  "type": "video_url",
  "video_url": {"url": "..."},
  "fps": 4.0,
  "media_resolution": "default"
}
```

The one canonical-audio fallback uses
`{"type":"input_audio","input_audio":{"data":"..."}}`; it never uses
`audio_url`. Every primary, fallback, and structure-repair request sends
`extra_body={"thinking":{"type":"disabled"}}`. MiMo reasoning is
intentionally disabled because this dataset path needs deterministic structured
annotation rather than agentic chain-of-thought, while avoiding unnecessary
tokens and latency. Nonzero reported reasoning tokens are retained as runtime
diagnostics. A token-limit finish reason fails closed and never enters the
text-only structure repair.

## Server commands

Run the known-case manifest without calling the API first:

```bash
"$R2V_PYTHON" tools/run_h3_mimo25_av_reconcile.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --known-case-pilot \
  --media-root /mnt/workspace \
  --media-mode base64 \
  --dry-run
```

Run the known cases, then the complete current inventory:

```bash
export MIMO_API_KEY='...'

"$R2V_PYTHON" tools/run_h3_mimo25_av_reconcile.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --known-case-pilot \
  --model mimo-v2.5 \
  --fps 4 \
  --media-resolution default \
  --media-root /mnt/workspace \
  --media-mode base64

"$R2V_PYTHON" tools/run_h3_mimo25_av_reconcile.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --model mimo-v2.5 \
  --fps 4 \
  --media-resolution default \
  --media-root /mnt/workspace \
  --media-mode base64 \
  --overwrite
```

Materialize and review without another model call:

```bash
"$R2V_PYTHON" tools/materialize_h3_mimo25_shadow.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --overwrite

"$R2V_PYTHON" tools/serve_h3_mimo25_review.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --legacy-qwen38-root "$QWEN38_ROOT" \
  --host 127.0.0.1 \
  --port 8768
```

Open `http://127.0.0.1:8768`. Review annotations are fingerprint-bound and are
stored under `mimo25_h3_shadow_v2/human_review/`.
