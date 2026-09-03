# H3 MiMo-V2.5 AV Shadow

This experimental path is additive and read-only with respect to the current JEA
production stages. It writes only `mimo25_av_reconcile_v4/` and
`mimo25_h3_shadow_v4/` under the Audio production root.

## Authority contract

- DiariZen owns exact speech segment times and sample boundaries.
- Qwen3-ASR owns exact transcript text and language.
- frozen Visual V3 references own entity inventory, order, and image content.
- LR-ASD, source clusters, and current entity bindings are proposals.
- MiMo-V2.5 reconciles clip-local speaker groups and visible entities, classifies
  speech audiovisual presentation, describes AV-grounded non-speech audio, and
  writes an H3 visual/temporal draft.
- the deterministic materializer owns `Sx`, Subject and Audio references, exact
  dialogue, and final H3 formatting.

MiMo never splits or deletes a DiariZen segment. Multiple vocal events can mark
a segment as requiring acoustic refinement, but the authoritative segment and
Qwen3-ASR text remain present. The default inventory scope is
`canonical_visual_target_inventory` with `canonical_wide_coverage=true` and
exact Visual, canonical Audio, DiariZen, Final H3, and segment-evidence coverage.
An explicit case manifest switches the scope to `explicit_case_subset` and sets
canonical-wide coverage false.

Speaker identity, visible-entity binding, and speech presentation are separate
facts. A voice matching a visible character does not establish visible speech.
Only `onscreen_spoken` with MiMo-observed `visible_lip_motion` may materialize as
`<Subject N> (Sx) says, ...`. `offscreen_spoken`, `voice_over`,
`message_voice_over`, and `device_playback` preserve the authoritative speech
and speaker group but never create a visible mouth-speaking action. An uncertain
presentation similarly removes visible-entity binding rather than guessing.

Each clip must have exactly one `pair_type="canonical"` Final H3 sample. That
sample is the target-observation representative and uses the `visual_only`
conditioning contract. Optional `in_pair` and `cross_pair` variants must carry
identical target video, full audio, Visual references, instruction, and
full-clip Audio semantics; all variant sample IDs remain in provenance, but
they do not create additional MiMo model jobs.

Prompt, policy, annotation schema, and materializer versions are:

- `h3_mimo25_unified_av_reconcile_v4`
- `h3_mimo25_av_authority_contract_v4`
- `r2v.h3.mimo25_av_annotation.4`
- `r2v.h3.mimo25_backend.4`
- `h3_mimo25_materializer_v4`
- `r2v.h3.mimo25_inventory.3`
- `r2v.h3.mimo25_record.4`
- `r2v.h3.mimo25_summary.4`
- `r2v.h3.mimo25_failure.4`
- `r2v.h3.mimo25_raw_response.4`
- `r2v.h3.mimo25_h3_shadow.4`
- `r2v.h3.mimo25_h3_shadow_summary.4`

The OpenAI-compatible client defaults to model `mimo-v2.5`, video FPS 4,
`media_resolution=default`, disabled thinking, JSON-object output,
temperature 0.2, and 16384 completion tokens. Base64 is the pilot default.
Payloads and API keys are never persisted. Explicitly reported zero video
tokens fail closed. If primary embedded-audio tokens are exactly zero, one
request retries with canonical full audio; explicitly reported zero audio
tokens on that request also fail closed. Unavailable usage details produce
warnings and do not trigger a blind retry.

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
`audio_url`. Every primary, fallback, and full-AV-recheck request sends
`extra_body={"thinking":{"type":"disabled"}}`. MiMo reasoning is
intentionally disabled because this dataset path needs deterministic structured
annotation rather than agentic chain-of-thought, while avoiding unnecessary
tokens and latency. Nonzero reported reasoning tokens are retained as runtime
diagnostics. Only `stop` is an explicitly successful finish reason; unavailable
finish reason is retained as a warning, while token limits and every other
explicit non-stop reason fail closed without a semantic recheck.

After the authoritative primary or canonical-audio-fallback response is
selected, parse or semantic validation failure may trigger at most one full AV
recheck with the same references, target video, and selected audio modality.
There is no text-only semantic repair. The v4 annotation assigns chronological
`ae1`, `ae2`, ... IDs to non-speech Audio events and requires each event exactly
once as `[[audio_event:aeN]]` in a shot. The deterministic MiMo materializer
replaces those placeholders with the validated event descriptions before the
existing exact speech-placeholder materialization, so final H3 text cannot
silently omit accepted Audio events.

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
stored under `mimo25_h3_shadow_v4/human_review/`.
