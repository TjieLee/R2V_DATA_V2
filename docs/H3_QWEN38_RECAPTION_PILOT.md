# H3 Qwen3.8 Recaption Pilot

## Purpose

`qwen38_h3_recaption_v1` is a read-only sidecar pilot. It observes a target
video, every frozen Visual reference image, rough Visual text, and normalized
upstream Audio **text** facts, then proposes a MiniMax H3 Ref2VA prompt for
manual review.

Qwen3.8 is primarily the rich visual/video recaptioner. It does not perform
ASR, diarization, speaker identification, binding correction, reference
selection, or Audio waveform analysis. No raw audio is sent to Qwen3.8.

The pilot writes only:

```text
<AUDIO_PRODUCTION_ROOT>/qwen38_h3_recaption_v1/
  manifest.jsonl
  records.jsonl
  summary.json
  raw_responses/
  review.html
```

It never modifies `h3/samples.jsonl` and is not yet authoritative production
H3 output. The current source remains the available final H3 sample inventory.
GitHub issue #11, which tracks widening that inventory to canonical full-clip
coverage, remains deliberately deferred.

## Official Contract

The implementation was written against the current official
`MiniMax-AI/MiniMax-H3` prompt-writing materials:

```text
docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md
skills/h3-prompt-writing/SKILL.md
skills/h3-prompt-writing/references/ref-en.txt
skills/h3-prompt-writing/references/base-en.txt
```

The two `docs/` files remain available in the official Hugging Face model
repository. The current GitHub repository publishes the same guides under the
skill's `references/` directory. The complete Ref2VA example in `ref-en.txt`
was inspected, but is not copied here.

The local system-prompt version is:

```text
h3_qwen38_ref2va_recaption_v1
```

The deterministic renderer publishes these exact sections in this exact order:

```text
subject_definitions
summary
retention_analysis
detailed_description
overall_soundscape
non_diegetic_music
```

The model returns a strict structured object. `audio_fact_audit` is retained as
diagnostic metadata and is not rendered into the H3 prompt.

## Frozen References

- The target video is observation-only and never becomes `<Video N>`.
- Every frozen Visual reference preserves its canonical `image_index` as
  `<Picture image_index>`. Images are not dropped, reordered, selected, or
  renumbered.
- Unique entity references become `<Subject N>` in first-Picture order.
- Attribute references become separate owner-bound Subject units.
- A background reference becomes an environment Subject unit.
- `(S1)`, `(S2)`, and later IDs are assigned by first actual speech event using
  stable `speaker_cluster_id` continuity.
- Bound speech retains its precomputed Subject; unbound speech keeps `(Sx)` and
  remains unbound.
- Picture, Video, and Audio numbering are independent.

The official Ref2VA limits are enforced without silent filtering: at most nine
images, three videos, three audio clips, and twelve total reference files.
This pilot currently defines zero H3 Video reference assets.

## Conditioning Variants

| Variant | Summary prefix | Audio retention |
| --- | --- | --- |
| `visual_only` | `[reference generation]` | no Audio label |
| `target_voice_reference` | `[reference generation + audio reference]` | `reference` |
| `cross_voice_reference` | `[reference generation + audio reference]` | `reference` |
| `full_audio_reuse` | `[reference generation + audio reuse]` | `fully_copy` |

Voice-reference and full-audio-reuse conditions are never combined by the
default contract. Donor dialogue is not target dialogue. Locked target ASR text
is the only dialogue authority.

## Audio Authority

Speech facts come from the current final H3 sample and preserve exact text,
language, time, cluster, and final entity binding. Optional full-clip semantics
must be supplied explicitly with `--audio-semantics-root`; the pilot never
guesses a root or reruns Dots3.

When semantic records are available, temporal non-speech events, delivery,
soundscape, and non-diegetic music are projected through current repository
schemas. Missing semantics set `audio_grounding_complete=false` and do not
create replacement sounds. Qwen has no raw-audio perception. It may only
generalize an obviously incorrect visual source attribution while preserving
the audible event; it cannot delete an event or alter ASR, speakers, bindings,
or timestamps.

Every locked dialogue block must appear exactly once:

```text
<d>[Language] EXACT_ASR_TEXT</d>
```

One structured repair is allowed. Unknown labels, `<Video N>`, changed or
duplicated dialogue, incompatible task/retention markers, and invented Audio
facts fail closed. The official 350-500-word generation guidance is recorded as
a warning rather than an arbitrary hard gate.

## One-Time Server Setup

The checkpoint is immutable input:

```bash
export QWEN38_CHECKPOINT=/mnt/workspace/guocong/model/Qwen/Qwen3.8-Flash-Next
export QWEN38_MODEL=Qwen/Qwen3.8-Flash-Next
export QWEN38_PORT=8000
export QWEN38_TP_SIZE=8
export QWEN38_MEDIA_ROOT=/mnt/workspace

export HF_HOME=/mnt/workspace/litengjie/data/audio_deps/qwen38-cache/huggingface
export XDG_CACHE_HOME=/mnt/workspace/litengjie/data/audio_deps/qwen38-cache/xdg
export VLLM_CACHE_ROOT=/mnt/workspace/litengjie/data/audio_deps/qwen38-cache/vllm
export QWEN38_LOG_ROOT=/mnt/workspace/litengjie/data/audio_deps/qwen38-logs

mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$QWEN38_LOG_ROOT"

vllm serve "$QWEN38_CHECKPOINT" \
  --served-model-name "$QWEN38_MODEL" \
  --host 127.0.0.1 \
  --port "$QWEN38_PORT" \
  --tensor-parallel-size "$QWEN38_TP_SIZE" \
  --allowed-local-media-path "$QWEN38_MEDIA_ROOT" \
  >"$QWEN38_LOG_ROOT/server.log" 2>&1
```

This command does not edit the checkpoint. Video sampling is sent per request
through `mm_processor_kwargs`; no processor config is written into the model
directory.

## Random200 Pilot

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2

export R2V_PYTHON=/mnt/workspace/litengjie/data/audio_deps/r2v-audio-env/bin/python
export AUDIO_PRODUCTION_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/random200/random200-src10000-19999-seed20260831-20260831-124415
export QWEN38_CASES=/mnt/workspace/litengjie/data/r2v_audio_runs/random200/qwen38-recaption-cases.jsonl
export QWEN38_OUTPUT="$AUDIO_PRODUCTION_ROOT/qwen38_h3_recaption_v1"

"$R2V_PYTHON" tools/run_h3_qwen38_recaption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --prepare-manifest "$QWEN38_CASES" \
  --manifest-size 5 \
  --conditioning-variant visual_only
```

Review and edit the explicit manifest before inference. Then run:

```bash
"$R2V_PYTHON" tools/run_h3_qwen38_recaption.py \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --case-manifest "$QWEN38_CASES" \
  --base-url "http://127.0.0.1:${QWEN38_PORT}/v1" \
  --served-model-name "$QWEN38_MODEL" \
  --checkpoint-id "$QWEN38_CHECKPOINT" \
  --media-mode file \
  --media-root /mnt/workspace \
  --video-fps 4 \
  --max-tokens 8192
```

When a frozen full-clip Audio semantics root is available, add only the explicit
argument:

```text
--audio-semantics-root /absolute/path/to/specialized_audio_semantics
```

Do not infer that path from another stage.

## Review

```bash
python -m http.server 8765 --directory "$QWEN38_OUTPUT"
```

Open `http://127.0.0.1:8765/review.html`. The page shows the target video, all
Pictures in canonical order, deterministic contracts, rough Visual hints,
normalized Audio facts, rendered prompt, audit metadata, and warnings.
