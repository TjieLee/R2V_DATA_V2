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
h3_qwen38_ref2va_recaption_v3
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

Qwen now returns `r2v.h3.qwen38_recaption_draft.1`. Each shot contains a
`description_template` with exactly one chronological `[[speech_N]]`
placeholder per frozen speech fact. Qwen does not serialize `[Shot N]`, speaker
sources, or `<d>` dialogue. The deterministic `h3_qwen38_materializer_v1`
renders shot headers and replaces each placeholder with the pipeline-owned
bound `<Subject N> (Sx)` or unbound `(Sx)` source plus exact locked dialogue.
The published structured sections and H3 prompt contain only this materialized
final text; raw sidecars preserve the original model draft.

Missing, duplicate, unknown, or reordered placeholders fail draft validation.
Direct shot headers, speaker serialization, or dialogue in a model draft also
fail. The materialized result still passes the complete final H3 validator, so
reference, retention, speaker, exact-dialogue, and shot syntax gates remain in
force.

`audio_fact_audit` is restricted to the exact supplied non-speech fact IDs;
speech IDs are never audit entries. When upstream Audio grounding is
incomplete and supplies neither non-speech events nor soundscape/music hints,
the two rendered Audio fields use fixed conservative wording that says those
facts are not established. The model cannot infer room tone, ambience, music,
or other sounds from the target video's visible content.

One structured repair is allowed. Unknown labels, `<Video N>`, changed or
duplicated dialogue, incompatible task/retention markers, and invented Audio
facts fail closed. The official 350-500-word generation guidance is recorded as
a warning rather than an arbitrary hard gate.

## Qwen3.8 Server Runtime

Validated server runtime as of 2026-09-01 uses **SGLang**, not the earlier vLLM
prototype. The Qwen3.8 checkpoint is immutable input:

```text
checkpoint: /mnt/workspace/guocong/model/Qwen/Qwen3.8-Flash-Next
served model: Qwen/Qwen3.8-Flash-Next
SGLang env: /mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env
SGLang source: /mnt/workspace/litengjie/data/audio_deps/sglang-qwen38-src
Qwen3.8 SGLang PR: sgl-project/sglang#36497
pinned PR HEAD used for install: 78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385
```

The old `qwen38-vllm-env` and source checkout are not the production runtime for
this pilot. The vLLM route required a dedicated Qwen3.8 runtime/image and was
abandoned on this notebook because Docker/Podman/Apptainer/Singularity are not
available.

### Recreate the SGLang environment

Use an independent persistent environment; do not modify the repo `.venv` or
the known-working Dots3/vLLM environments.

```bash
export SGLANG_ENV=/mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env
export SGLANG_SRC=/mnt/workspace/litengjie/data/audio_deps/sglang-qwen38-src

uv venv --python 3.12 --seed "$SGLANG_ENV"

git clone https://github.com/sgl-project/sglang.git "$SGLANG_SRC"
cd "$SGLANG_SRC"
git fetch origin pull/36497/head
git checkout --detach FETCH_HEAD
git rev-parse HEAD
```

Expected pinned HEAD:

```text
78c5024e9d9f589dcb4deb7f4ba4fb23f7e85385
```

The source build probes optional Rust extensions by default. This server does
not provide `cargo`; Rust extensions are not required for the current Qwen3.8
CUDA inference path. Install with the repository-supported opt-out:

```bash
SGLANG_BUILD_RUST_EXTS=none \
uv pip install \
  --python "$SGLANG_ENV/bin/python" \
  -e python
```

Do not install a Rust toolchain only to satisfy editable-install metadata.

### Start Qwen3.8 in a fresh terminal

The current launch follows the Qwen3.8 SGLang recommended configuration, with
three task/machine-specific differences: TP is 8 instead of 4, context is
capped at 49,152 for the H3 pilot, and the service is bound to localhost.
Port choice is operational only.

```bash
export SGLANG_ENV=/mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env
export QWEN38_CHECKPOINT=/mnt/workspace/guocong/model/Qwen/Qwen3.8-Flash-Next
export QWEN38_MODEL=Qwen/Qwen3.8-Flash-Next
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd /mnt/workspace/litengjie/data

"$SGLANG_ENV/bin/sglang" serve \
  --model-path "$QWEN38_CHECKPOINT" \
  --served-model-name "$QWEN38_MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --tp 8 \
  --context-length 49152 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 \
  --linear-attn-prefill-backend flashinfer \
  --linear-attn-decode-backend flashinfer \
  --linear-attn-verify-backend triton \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --max-running-requests 96 \
  --reasoning-parser auto
```

For BF16 on CUDA, the documented Qwen3.8 SGLang path auto-enables PLE offload;
do not add an independent PLE override unless debugging an actual load/runtime
failure.

Do **not** add `--enable-mixed-chunk` to the current Qwen3.8 runtime. The
Qwen3.8/QSA path has had a mixed-chunk stability issue; the current operational
launch intentionally leaves it disabled.

### Non-thinking request contract

`--reasoning-parser auto` only parses reasoning when the model emits it; it does
not disable thinking. H3 recaption must explicitly request non-thinking mode.
The Qwen3.8 recommended non-thinking sampling contract is:

```text
temperature: 0.7
top_p: 0.80
top_k: 20
min_p: 0.0
presence_penalty: 1.5
repetition_penalty: 1.0
chat_template_kwargs.enable_thinking: false
```

Minimal API smoke test:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3.8-Flash-Next",
    "messages": [
      {"role": "user", "content": "Reply with exactly: OK"}
    ],
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {
      "enable_thinking": false
    },
    "max_tokens": 128
  }'
```

A prior text smoke test without `enable_thinking=false` returned separate
`reasoning_content`, proving thinking was active. For H3 recaption, require
`reasoning_content` to be null/absent and use only the final content/structured
response.

### Current adapter boundary

The recaption client uses the validated SGLang OpenAI-compatible request
contract. It sends target `video_url`, frozen-reference `image_url`, strict
`response_format=json_schema`, and the non-thinking sampling parameters above.
It does not send the former vLLM-only `mm_processor_kwargs` FPS override and
uses SGLang/Qwen3.8's default video processing behavior. The frozen
prompt/reference/audio authority rules above are unchanged.

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
  --base-url "http://127.0.0.1:8000/v1" \
  --served-model-name "$QWEN38_MODEL" \
  --checkpoint-id "$QWEN38_CHECKPOINT" \
  --media-mode file \
  --media-root /mnt/workspace \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repetition-penalty 1.0 \
  --max-tokens 8192
```

The non-thinking sampling contract is `temperature=0.7`, `top_p=0.8`,
`top_k=20`, `min_p=0.0`, `presence_penalty=1.5`,
`repetition_penalty=1.0`, and `enable_thinking=false`. The last four SGLang
extensions are transported through the OpenAI client's `extra_body` where
needed; no `mm_processor_kwargs` are sent.

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
