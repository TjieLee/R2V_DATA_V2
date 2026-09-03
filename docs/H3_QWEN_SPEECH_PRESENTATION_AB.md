# H3 Qwen Speech-Presentation A/B Shadow

This is a temporary, model-neutral shadow experiment. It uses the existing
Qwen Ref2VA visual draft contract while asking Qwen3.5 and Qwen3.8 to classify
how every authoritative speech segment is presented visually. It does not
replace MiMo-V2.5 and never writes production `h3/samples.jsonl`.

The case manifest is the inventory authority. The current comparison reuses the
existing `qwen35_vs_qwen38_ab12.jsonl` without resampling, but the implementation
accepts any non-empty explicit manifest. Cases must be canonical Final H3 samples
with `conditioning_variant="visual_only"`; in-pair and cross-pair cases fail
closed.

## Authority And Scope

- Qwen3-ASR text/language and DiariZen timing remain byte-exact facts.
- Existing speaker cluster to `Sx` assignment remains unchanged.
- The current visible-entity binding is a fallible proposal, not model truth.
- Only `onscreen_spoken` with `visible_lip_motion` may retain or assign a visible
  entity. Every other presentation removes the visible speaking relation in the
  shadow facts.
- The model receives target video, frozen reference images, and locked speech
  text. It receives no raw audio and makes no acoustic speaker-identity claim.
- The deterministic materializer owns exact dialogue and uses the same
  presentation wording as MiMo v4.

The presentation values are:

```text
onscreen_spoken
offscreen_spoken
voice_over
message_voice_over
device_playback
uncertain
```

The shadow is intended to test visible lip speech, absence of visible speech,
clear message voice-over context, and deterministic H3 insertion. It does not
perform acoustic speaker identity, full speaker regrouping, or authoritative
offscreen/voice-over/device source classification. Uncertain visual evidence
remains `uncertain`.

## Versions

```text
prompt:       h3_qwen_ref2va_speech_presentation_v1
policy:       h3_qwen_ref2va_speech_presentation_contract_v1
draft:        r2v.h3.qwen_speech_presentation_draft.1
backend:      r2v.h3.qwen_speech_presentation_backend.1
record:       r2v.h3.qwen_speech_presentation_record.1
summary:      r2v.h3.qwen_speech_presentation_summary.1
materializer: h3_qwen_speech_presentation_materializer_v1
```

The existing standalone Qwen3.8 and MiMo v4 version strings remain unchanged.

## Server Runs

Reuse the original A/B manifest and run the same harness once per model:

```bash
CASE_MANIFEST="$AUDIO_PRODUCTION_ROOT/qwen35_vs_qwen38_ab12.jsonl"

"$R2V_PYTHON" tools/run_h3_qwen_speech_presentation_recaption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --case-manifest "$CASE_MANIFEST" \
  --base-url "$QWEN35_BASE_URL" \
  --api-key "${QWEN35_API_KEY:-EMPTY}" \
  --served-model-name "$QWEN35_SERVED_MODEL_NAME" \
  --checkpoint-id "$QWEN35_CHECKPOINT_ID" \
  --media-mode file \
  --media-root /mnt/workspace \
  --output-root "$AUDIO_PRODUCTION_ROOT/qwen35_397b_h3_ab12_presentation_v1"

"$R2V_PYTHON" tools/run_h3_qwen_speech_presentation_recaption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --case-manifest "$CASE_MANIFEST" \
  --base-url "$QWEN38_BASE_URL" \
  --api-key "${QWEN38_API_KEY:-EMPTY}" \
  --served-model-name "$QWEN38_SERVED_MODEL_NAME" \
  --checkpoint-id "$QWEN38_CHECKPOINT_ID" \
  --media-mode file \
  --media-root /mnt/workspace \
  --output-root "$AUDIO_PRODUCTION_ROOT/qwen38_flash_h3_ab12_presentation_v1"
```

Build the static four-way review without another model call:

```bash
"$R2V_PYTHON" tools/build_h3_qwen_speech_presentation_ab_review.py \
  --qwen35-old-root "$AUDIO_PRODUCTION_ROOT/qwen35_397b_h3_ab12" \
  --qwen35-new-root "$AUDIO_PRODUCTION_ROOT/qwen35_397b_h3_ab12_presentation_v1" \
  --qwen38-old-root "$AUDIO_PRODUCTION_ROOT/qwen38_flash_h3_ab12" \
  --qwen38-new-root "$AUDIO_PRODUCTION_ROOT/qwen38_flash_h3_ab12_presentation_v1" \
  --output-root "$AUDIO_PRODUCTION_ROOT/qwen35_vs_qwen38_ab12_presentation_review"
```

Open `review.html` in each new model root for individual inspection, or the
combined root for the four-way old/new comparison. Review whether old false
visible-speaking relations disappear, clear message voice-over is recognized,
real visible speech remains entity-bound, uncertainty is not overused, and the
additional task does not degrade the visual recaption.
