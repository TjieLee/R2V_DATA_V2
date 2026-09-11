# H3 T2VA / TA2VA Server Runbook

This is the current server-side happy path for the frozen target-side no-visual-reference products.
It is intentionally short and operational; design details remain in `H3_T2VA_SHADOW.md` and
`H3_TA2VA_SHADOW.md`.

## Frozen state

Current target-side freeze:

- T2VA prompt: `h3_t2va_joint_av_v7`
- T2VA backend: `r2v.h3.t2va_mimo_backend.7`
- T2VA no-reference core: `r2v.h3.no_reference_av_core.6`
- TA2VA product schema: `r2v.h3.ta2va_shadow.2`
- TA2VA speaker profile: `h3_ta2va_speaker_profile_v2`
- R2VA audio finalizer reused unchanged: `h3_mimo25_audio_finalize_v6`

T2VA remains a three-section no-reference product and normally makes exactly two MiMo calls per
ready clip: semantic AV, then the frozen non-dialogue audio finalizer. TA2VA consumes the frozen
T2VA v7 core downstream; it does not rerun T2VA AV semantics or the audio finalizer.

TA2VA currently publishes only:

- `full_audio_reuse`: canonical full target audio, `fully_copy`, zero TA2VA model calls;
- `target_speech_reuse`: full-timeline per-Sx speech reuse plus optional music reuse,
  `partially_copy`, one speaker-profile call per eligible clip.

Cross-pair is intentionally deferred to GitHub Issue #13. Do not add cross-pair matching to this
path. Future cross-pair should use one frozen face-similarity donor mapping so donor visual and
speech/voice references remain identity-consistent.

## Server environment

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2

git fetch origin feature/h3-audio-jea-qwen3-v1
git switch feature/h3-audio-jea-qwen3-v1
git merge --ff-only origin/feature/h3-audio-jea-qwen3-v1

source .venv/bin/activate
source server_env.sh
unset PYTHONPATH
export R2V_PYTHON="$(command -v python)"
```

For the validated random20 cache:

```bash
export SHOT_MANIFEST=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl
export JEA_CLIPS_ROOT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped
export JEA_SOURCE_VIDEOS_ROOT=/mnt/workspace/public/dataset/jea-video/moive-183t-0808
export SHOT_INDEX_ROOT=/mnt/workspace/litengjie/data/t2va_shot_index

export AUDIO_PRODUCTION_ROOT=/mnt/workspace/litengjie/data/t2va_audio/random20-seed20260911-v2
export AUDIO_SHADOW_RUN_ID=random20-musicfirst-auk-v1
```

The resolved Audio policy is frozen:

```text
speech -> AuK Base
music  -> SAM Audio music_first
sfx    -> SAM Audio music_first
```

Do not rerun DiariZen, Qwen3-ASR, AuK or SAM when this named Audio run already has matching
resolved lineage. Missing or mismatched lineage must fail rather than silently fall back to a
legacy route.

## MiMo server

If the local MiMo endpoint is already serving, only verify it:

```bash
curl -fsS http://127.0.0.1:8092/v1/models
```

If it must be started:

```bash
export SGLANG_ENV=/mnt/workspace/litengjie/data/audio_deps/qwen38-sglang-env
export MIMO_CHECKPOINT=/mnt/workspace/public/pretrained/MiMo/MiMo-V2.5

nohup "$SGLANG_ENV/bin/sglang" serve \
  --model-path "$MIMO_CHECKPOINT" \
  --served-model-name mimo-v2.5 \
  --host 127.0.0.1 \
  --port 8092 \
  --trust-remote-code \
  --tp 8 \
  --dp 2 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --mm-enable-dp-encoder \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --context-length 131072 \
  --mem-fraction-static 0.65 \
  --chunked-prefill-size 16384 \
  --max-running-requests 8 \
  --reasoning-parser mimo \
  --tool-call-parser mimo \
  --constrained-json-disable-any-whitespace \
  --enable-deterministic-inference \
  > /tmp/mimo8092-t2va.log 2>&1 &

until curl -fsS http://127.0.0.1:8092/v1/models >/dev/null 2>&1; do sleep 5; done
```

## Run T2VA v7

Use a fresh T2VA run ID whenever the T2VA prompt/backend/core contract changes.
The validated random20 command is:

```bash
export T2VA_RUN_ID=t2va-random20-v7

"$R2V_PYTHON" tools/run_h3_t2va_shadow.py \
  --shot-manifest "$SHOT_MANIFEST" \
  --clips-root "$JEA_CLIPS_ROOT" \
  --source-videos-root "$JEA_SOURCE_VIDEOS_ROOT" \
  --shot-index-root "$SHOT_INDEX_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --audio-shadow-run-id "$AUDIO_SHADOW_RUN_ID" \
  --t2va-run-id "$T2VA_RUN_ID" \
  --sample-size 20 \
  --sample-seed 20260911 \
  --base-url http://127.0.0.1:8092/v1 \
  --transport sglang \
  --model mimo-v2.5 \
  --media-root /mnt/workspace \
  --max-completion-tokens 32768
```

Inspect only the direct outputs needed for QA:

```bash
export T2VA_ROOT="$AUDIO_PRODUCTION_ROOT/t2va_shadow_v1/runs/$T2VA_RUN_ID"

cat "$T2VA_ROOT/summary.json"

jq -r '
  select(.status!="ready")
  | [.clip_uid, .status, .failure_reason]
  | @tsv
' "$T2VA_ROOT/records.jsonl"
```

A fully ready random20 has 20 records and 40 total model calls. T2VA v7 additionally stores the
internal summary and per-segment waveform-ownership evidence required by TA2VA, but its public
prompt remains exactly three sections.

## Run TA2VA .2 from the frozen T2VA v7 core

Do not rerun T2VA. A model-free preflight is optional:

```bash
export TA2VA_RUN_ID=target-ta2va-random20-v2-final

"$R2V_PYTHON" tools/run_h3_ta2va_shadow.py \
  --t2va-root "$T2VA_ROOT" \
  --ta2va-run-id "$TA2VA_RUN_ID" \
  --base-url http://127.0.0.1:8092/v1 \
  --transport sglang \
  --model mimo-v2.5 \
  --media-root /mnt/workspace \
  --allow-unverified \
  --dry-run
```

Then remove only `--dry-run`:

```bash
"$R2V_PYTHON" tools/run_h3_ta2va_shadow.py \
  --t2va-root "$T2VA_ROOT" \
  --ta2va-run-id "$TA2VA_RUN_ID" \
  --base-url http://127.0.0.1:8092/v1 \
  --transport sglang \
  --model mimo-v2.5 \
  --media-root /mnt/workspace \
  --allow-unverified
```

TA2VA uses the authoritative raw source samples to put each Sx's eligible speech segments back at
their original target positions in a target-duration zero/silence buffer. It never concatenates
segments or reconstructs placement from floating-point timestamps. Any unsafe ownership segment
blocks the whole Sx from speech reuse using the shared `speaker_ownership_reasons()` policy;
`same_speaker_nonlexical` remains the narrow allowed exception.

The profile call is audio-only and fixed to already-frozen Sx targets. SGLang uses request-specific
strict slots and positional Sx constants; there is one call for all selected Sx in the clip, no
retry and no fallback profile. The profile may describe supported acoustic/prosodic traits only.

The final six-section TA2VA prompt has no Picture/Subject/Video references. Its summary task prefix
is Audio-only:

```text
summary:
[audio reuse] ...
```

The shared Ref2VA/RA2VA `[reference generation + ...]` prefix is not used by TA2VA.

Inspect the run:

```bash
export TA2VA_ROOT="$AUDIO_PRODUCTION_ROOT/ta2va_shadow_v1/runs/$TA2VA_RUN_ID"

cat "$TA2VA_ROOT/summary.json"

jq -r '
  select(.status!="ready")
  | [.clip_uid, .variant, .status, .failure_reason]
  | @tsv
' "$TA2VA_ROOT/records.jsonl"

jq '.reuse_exclusions' "$TA2VA_ROOT/summary.json"

jq -r '
  [.clip_uid, .variant, .status, .model_call_count,
   ([.audio_references[].kind] | join(","))]
  | @tsv
' "$TA2VA_ROOT/records.jsonl"
```

## Validated random20 acceptance

The real random20 TA2VA .2 acceptance run produced:

```text
clip_count              20
product_count           37
ready_count             37
failed_count            0
model_call_count        17
full_audio_reuse        20
speaker_speech_reuse    20
music_reuse             10
target_speech_reuse     17
```

One clip, `f3c9b72d7be7b0fda85d152d`, intentionally produced no target-speech product because all
three S1 segments had `secondary_vocal_activity`; the full-audio product remained ready. This is an
ownership exclusion, not a model failure.

The acceptance run named `target-ta2va-random20-v2` predates the final deterministic summary-prefix
cleanup. Its model/profile/PCM results remain valid evidence, but its stored prompt text still says
`[reference generation + audio reuse]`. Final publication under the current branch must use a fresh
TA2VA run ID so prompts materialize with `[audio reuse]`. T2VA v7 does not need to be rerun.

## TA2VA static QA

Generate QA from the current TA2VA run:

```bash
"$R2V_PYTHON" tools/build_h3_ta2va_qa.py \
  --ta2va-root "$TA2VA_ROOT" \
  --media-root /mnt/workspace \
  --media-base-url http://127.0.0.1:8776 \
  --overwrite
```

Serve `/mnt/workspace` in another terminal:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate
python -m http.server 8776 --bind 127.0.0.1 --directory /mnt/workspace
```

For the validated random20 root, the browser path shape is:

```text
http://127.0.0.1:8776/litengjie/data/t2va_audio/random20-seed20260911-v2/ta2va_shadow_v1/runs/<TA2VA_RUN_ID>/qa/review.html
```

Manual QA should focus on:

1. each Sx reuse track is silent outside its owned segments and preserves original target timing;
2. voice characteristics match the audible speaker and contain no identity/psychology/cross-Sx claims;
3. `<Audio N>` ↔ `(Sx)` relations are correct and no `<Picture N>` / `<Subject N>` / `<Video N>` appears;
4. exact ASR, Sx, shot ordering and timestamps remain unchanged from frozen T2VA;
5. published music reuse matches actual audience-only BGM and obvious music is not missed.

## Freeze / change policy

The current target-side freeze is T2VA v7 + TA2VA .2. Do not change these paths for unrelated
cleanup after QA acceptance. Any future semantic contract change must use a new run ID and the
appropriate schema/prompt/backend version bump.

Do not mutate frozen R2VA/RA2VA production artifacts. LR-ASD remains supporting evidence only, not
a hard eligibility gate. `multiple_speakers` / unsafe waveform ownership remains a hard exclusion
for identity-specific speech reuse.
