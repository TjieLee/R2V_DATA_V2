# H3 Audio Server Runbook

Last updated: 2026-09-07

This is the server operating runbook for Audio/H3 development on
`feature/h3-audio-jea-qwen3-v1`. It complements the Visual-focused
`docs/SERVER_ENVIRONMENT_RUNBOOK.md`; do not copy Visual SAM3/Boogu
`PYTHONPATH` or GPU assumptions into Audio/H3 jobs.

Always verify the live branch and HEAD before operating the server. The named
SAM Audio shadow namespace was implemented at
`51d4b3c50ef62870a06f2ac4db4df0da444fd9bc` (`Add named SAM Audio shadow
runs`). Later documentation-only commits may advance HEAD without changing that
code baseline.

## Repository and current pilot roots

```text
repository:
  /mnt/workspace/litengjie/data/R2V_DATA_V2

main R2V Python:
  /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python

current random200 Audio production root:
  /mnt/workspace/litengjie/data/r2v_audio_runs/random200/random200-src10000-19999-seed20260831-20260831-124415

validated one-clip legacy/default SAM shadow root:
  <audio-production-root>/sam_audio_stem_shadow_v1/

named pilot root example:
  <audio-production-root>/sam_audio_stem_shadow_v1/runs/random10-v1/
```

The validated one-clip smoke under the legacy/default root is retained as a
regression baseline. New pilots should use `--shadow-run-id`; do not overwrite
the default smoke merely to change the case inventory.

The current named run uses these owned stages:

```text
sam_audio_stem_shadow_v1/runs/<shadow-run-id>/
  separation/
  diarization/
  asr/
  mimo_reconcile_av_rawstems_sound_partition/
  references/
```

All commands must receive the same run ID and ordered case manifest. Hash
and absolute-path lineage is run-local; copied artifacts from another named run
or from the legacy/default root fail closed.

## Safe repository update

Preserve server-local and untracked files. Do not use `git clean`, `git reset
--hard`, rebase, or force push.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
git fetch origin feature/h3-audio-jea-qwen3-v1
git switch feature/h3-audio-jea-qwen3-v1
git merge --ff-only origin/feature/h3-audio-jea-qwen3-v1
git status --short
git rev-parse HEAD
```

Known server-local files such as `server_env.sh`, model logs, local configs, and
pilot artifacts must not be deleted or committed merely to make the tree clean.

## Common Audio/H3 shell initialization

Use the existing main environment for orchestration:

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
source .venv/bin/activate

# server_env.sh is server-local/untracked. Validate and source the existing file;
# never recreate or commit it merely from this document.
bash -n server_env.sh
source server_env.sh

export R2V_PYTHON="$(command -v python)"
test "$R2V_PYTHON" = /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python
```

Audio/H3 scripts insert the repository root themselves. Do not export the SAM
Audio dependency overlay globally. If the shell inherited an unrelated Visual
`PYTHONPATH`, clear it before the Audio shadow sequence unless a specific stage
runbook says otherwise:

```bash
unset PYTHONPATH
```

## SAM Audio isolated dependency overlay

SAM Audio uses the main R2V interpreter plus an invocation-local supplemental
`PYTHONPATH`. Do not install SAM-only dependencies into the main `.venv`, and do
not replace its Torch/NumPy/Torchaudio stack.

```bash
export SAM_DEPS_ROOT=/mnt/workspace/litengjie/data/audio_deps
export SAM_AUDIO_CODE_ROOT="$SAM_DEPS_ROOT/sam-audio-src"
export PERCEPTION_MODELS_ROOT="$SAM_DEPS_ROOT/perception-models-src"
export DACVAE_ROOT="$SAM_DEPS_ROOT/dacvae-src"
export SAM_AUDIO_PYDEPS="$SAM_DEPS_ROOT/sam-audio-pydeps"
export SAM_AUDIO_RUNTIME_PYTHONPATH="$SAM_AUDIO_CODE_ROOT:$PERCEPTION_MODELS_ROOT:$DACVAE_ROOT:$SAM_AUDIO_PYDEPS"

export SAM_AUDIO_MODEL_PATH=/mnt/workspace/public/pretrained/Facebook/sam-audio-large-tv
export SAM_AUDIO_MODEL_NAME=facebook/sam-audio-large-tv
export SAM_AUDIO_T5_BASE_PATH=/mnt/workspace/public/pretrained/google/t5-base
```

The validated SAM source commit is
`bb4c6999d2677c7402360e426afc01ddfad6dce0`. Model/config/T5 assets are
local-files-only. `/mnt/workspace/public/pretrained/**` is read-only.

Run SAM separation with a one-command overlay so later stages do not inherit
SAM packages:

```bash
PYTHONPATH="$SAM_AUDIO_RUNTIME_PYTHONPATH" \
"$R2V_PYTHON" tools/run_h3_sam_audio_stem_shadow.py ...
```

Do not replace this with a persistent `export PYTHONPATH=...` across DiariZen,
Qwen3-ASR, or MiMo stages.

## DiariZen isolated worker

The shadow DiariZen runner is launched by `R2V_PYTHON`; its persistent worker
uses the existing isolated DiariZen runtime:

```bash
export DIARIZEN_PYTHON=/mnt/workspace/litengjie/data/audio_deps/diarizen-venv/bin/python
export DIARIZEN_CODE_ROOT=/mnt/workspace/litengjie/data/audio_deps/DiariZen
export DIARIZEN_MODEL_PATH=/mnt/workspace/litengjie/data/audio_deps/diarizen-model-cache
export DIARIZEN_MODEL_IDENTIFIER=BUT-FIT/diarizen-wavlm-large-s80-md-v2
export DIARIZEN_DEVICE=cuda:0
export DIARIZEN_TIMEOUT_SECONDS=900
```

Do not install DiariZen into the main `.venv`.

## Qwen3-ASR isolated worker

The Qwen3-ASR package remains isolated:

```bash
export QWEN3_ASR_ENV=/mnt/workspace/litengjie/data/audio_deps/qwen3-asr-venv
export QWEN3_ASR_MODEL_PATH=/mnt/workspace/public/pretrained/Qwen/Qwen3-ASR-1.7B
export QWEN3_ASR_DEVICE=cuda:0
export QWEN3_ASR_DTYPE=bfloat16
export QWEN3_ASR_MAX_INFERENCE_BATCH_SIZE=1
```

Important distinction:

- production `tools/run_h3_qwen3_asr.py` may be launched directly with
  `$QWEN3_ASR_ENV/bin/python` as documented in `docs/H3_QWEN3_ASR.md`;
- shadow `tools/run_h3_stem_qwen3_asr_shadow.py` must be launched with
  `$R2V_PYTHON`; it performs lineage/schema work in the main environment and
  starts a persistent `$QWEN3_ASR_ENV/bin/python` worker internally.

Do not launch the shadow runner with the Qwen interpreter and do not install
`qwen-asr` or OpenCV into the other environment to paper over import errors.
The worker clears inherited `PYTHONPATH`, loads Qwen once, and stays
local-files-only.

## MiMo endpoint

The validated stem-facts and reconcile pilots use the existing OpenAI-compatible
MiMo service:

```text
model:    mimo-v2.5
endpoint: http://127.0.0.1:8092/v1
```

The main R2V interpreter calls this endpoint. `MIMO_API_KEY` remains a
server-local environment value and must not be committed.

Optional health check:

```bash
curl -s http://127.0.0.1:8092/v1/models | head -c 1000
echo
```

## Named random10 shadow pilot

The current suggested next pilot is `random10-v1`. The existing ordered random10
manifest used during development has been `/tmp/mimo-v20-random10.json`; `/tmp`
is volatile, so verify that file before use rather than assuming it survived a
server restart.

```bash
export AUDIO_PRODUCTION_ROOT=/mnt/workspace/litengjie/data/r2v_audio_runs/random200/random200-src10000-19999-seed20260831-20260831-124415
export CASE_MANIFEST=/tmp/mimo-v20-random10.json
export SHADOW_RUN_ID=random10-v1

test -f "$CASE_MANIFEST"

RUN_ARGS=(
  --audio-production-root "$AUDIO_PRODUCTION_ROOT"
  --shadow-run-id "$SHADOW_RUN_ID"
  --case-manifest "$CASE_MANIFEST"
  --sam-route music_first
)
```

First run dry-runs for the selected inventory where useful. A new named run must
not need `--overwrite`.

### 1. SAM separation

```bash
PYTHONPATH="$SAM_AUDIO_RUNTIME_PYTHONPATH" \
"$R2V_PYTHON" tools/run_h3_sam_audio_stem_shadow.py "${RUN_ARGS[@]}" \
  --sam-audio-code-root "$SAM_AUDIO_CODE_ROOT" \
  --sam-audio-model-path "$SAM_AUDIO_MODEL_PATH" \
  --sam-audio-model-name "$SAM_AUDIO_MODEL_NAME" \
  --sam-audio-t5-base-path "$SAM_AUDIO_T5_BASE_PATH" \
  --sam-reranking-candidates 1
```

### 2. Speech-stem DiariZen

```bash
"$R2V_PYTHON" tools/run_h3_stem_diarization_shadow.py "${RUN_ARGS[@]}" \
  --allow-unverified
```

### 3. Speech-stem Qwen3-ASR

```bash
"$R2V_PYTHON" tools/run_h3_stem_qwen3_asr_shadow.py "${RUN_ARGS[@]}" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --allow-unverified
```

### 4. One original AV plus raw stems, then text-only sound partition

After separation, DiariZen and ASR, run this entry directly. Do not run
`run_h3_mimo25_stem_facts_shadow.py`; neither `mimo_stem_facts/` nor three
stem-video proxies are inputs.

```bash
"$R2V_PYTHON" tools/run_h3_mimo25_stem_reconcile_shadow.py "${RUN_ARGS[@]}" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --model mimo-v2.5 \
  --base-url http://127.0.0.1:8092/v1 \
  --max-completion-tokens 32768 \
  --allow-unverified
```

The first user content contains exactly one original target video with embedded
audio, existing reference images, and two `audio_url` items read from the selected
separation record: proposed music and SFX. They share clip time zero, may contain
leakage, and are observation evidence rather than H3 conditioning assets. There
is no speech-stem input, duplicate canonical full audio, stem semantic JSON, or
media repackaging. Original AV remains authoritative.

The first annotation keeps speaker/AV grounding and natural
`style_opening` / `shot1_caption`, and emits one `sound_description` instead of
global soundscape/music statuses or timed-event inventories. Existing speaker
normalization and identity-product exclusions remain intact.

The second request contains only a short system instruction and the first
`sound_description` as user text. It has no media, ASR, references, ICL, previous
messages or caption. Disabled thinking, temperature 0, 1024 completion tokens,
and a strict two-string JSON schema produce `overall_soundscape` and
`non_diegetic_music`. The partition is not a semantic audit or a remedy for
sounds missed in the first observation. Unsupported fields use `N/A` as a
placeholder, not a claim of confirmed silence. No content-keyword QC is applied.

Both stages make exactly one request attempt: no SDK retry, HTTP retry,
zero-audio fallback, full-AV recheck or third repair call. Errors preserve raw
output and diagnostics and later clips continue. A parseable
`sound_description` may be partitioned even when AV grounding or caption
format failed; that does not authorize identity-specific products.

First-stage defaults remain `--thinking disabled --icl official_ref2va_v1`,
temperature 0.0. The official opening + Shot 1 ICL
(`h3_official_ref2va_detailed_shot1_v2`) is unchanged. First-stage SGLang still
uses `use_audio_in_video=true`; the text stage does not send that flag.

Dialogue is model-authored. Consecutive same-speaker ASR segments may share one
natural `<d>[Language] ...</d>` block. Code checks paired tags, language markers,
allowed labels and a valid `(Sx)` in each generated vocal-event lead-in; it does
not count/zip against ASR segments, rewrite text, or infer binding from syntax.
Upstream ASR files remain immutable and are shown separately for review.
Detailed description is exactly
`style_opening + "\n[Shot 1] " + shot1_caption`; final sound sections come only
from the text partition. Six-section Ref2VA output and frozen reference ownership
remain unchanged. Raw caption stays visible when final materialization is blocked.

Current versions: prompt v29, annotation .19, backend .32, materializer v22,
authority v17 unchanged; raw-stem reconcile records .8, summary .10, policy v3.
New output is `mimo_reconcile_av_rawstems_sound_partition/`. Old experiments
are not migrated or rewritten. Normal random10 request counts are 10 AV and
10 text calls; counts are not quality or latency measurements.

**Deployment boundary:** the original joint-AV runtime was validated earlier.
The one-video/two-auxiliary-audio combination has only fake-client coverage in
this patch, not a real server run. Aggregate `audio_tokens > 0` cannot prove
that all three audio inputs were consumed. Unsupported combinations must retain
the actual error, never silently drop an audio track and resend the video.

### 5. Optional stem-native primary voice references

```bash
"$R2V_PYTHON" tools/export_h3_sam_audio_stem_references.py "${RUN_ARGS[@]}" \
  --primary-voice-root "$AUDIO_PRODUCTION_ROOT/primary_voice" \
  --allow-unverified
```

The reference exporter reuses only already accepted production primary-voice
intervals. It cuts the same exact sample range from the selected speech stem; it
does not create a new voice eligibility rule.

## Expected named-run outputs

```bash
export SHADOW_ROOT="$AUDIO_PRODUCTION_ROOT/sam_audio_stem_shadow_v1/runs/$SHADOW_RUN_ID"

find "$SHADOW_ROOT" -maxdepth 2 -type f | sort
cat "$SHADOW_ROOT/separation/summary.json"
cat "$SHADOW_ROOT/diarization/stem_provenance.json"
cat "$SHADOW_ROOT/asr/summary.json"
cat "$SHADOW_ROOT/mimo_reconcile_av_rawstems_sound_partition/summary.json"
cat "$SHADOW_ROOT/references/references.jsonl"
```

Expected stage ownership is strictly run-local. If an upstream stage is
intentionally replaced, all already-published downstream stages become stale by
lineage and must be inspected before any explicit replacement. Prefer a new run
ID for a new case inventory or algorithm experiment.

## Static manual QA sidecar

Inspect label quality without running any model or changing any source stage.
The explicit output below is outside production inputs and preserves old QA pages:

```bash
export QA_ROOT="$AUDIO_PRODUCTION_ROOT/../h3-audio-qa/$SHADOW_RUN_ID-av-rawstems-sound-partition"
"$R2V_PYTHON" tools/build_h3_audio_shadow_qa.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --shadow-run-id "$SHADOW_RUN_ID" \
  --case-manifest "$CASE_MANIFEST" \
  --output-root "$QA_ROOT"
"$R2V_PYTHON" -m http.server 8774 --bind 127.0.0.1 --directory "$QA_ROOT"

```

Forward port 8774 through the existing SSH connection and open
`http://127.0.0.1:8774/review.html` locally. Serve only the QA directory, not the
whole production tree. No POST endpoint or model environment is involved.
The page shows original target AV, frozen references, unchanged production
comparison text, shadow DiariZen/ASR, and optional stem audio players.
It reads the new reconcile directory without requiring stem-facts artifacts.
First raw caption and sound description remain visible for failed records;
text partition result/raw/error and both call counts are shown separately.

The exact final six-section prompt uses the existing materializer v22 path
when AV validation and partitioning permit it. Each source conditioning variant
remains separate. Multi-speaker attribution still blocks identity-specific
materialization, and blocked variants expose issues instead of best-effort text.
No voice recovery, media generation, H3 publication, or model call runs in QA.
Model prose is not overwritten by ASR. Rendered text and current records remain
fingerprint-bound, so changed output invalidates stale annotations.
The `better/same/worse` and issue-label review workflow is unchanged.

The builder validates existing complete stage metadata and lineage, including
per-clip failure records. Output is `review.html`, `data.json`, and `media/`
containing only single-file symlinks to provenance-checked video/image/stem
assets. No production media is copied or changed. Relative frontend URLs work
with a static server that follows these links; moving this directory to another
machine requires the same source paths. Browser codec support and server access
to the links still need checking on Linux. No media transcode is performed.

Omit the run ID for the default smoke's independent `qa/` folder. An optional
`--output-root` may select a separate directory outside all input roots.
An existing output requires `--overwrite` and must be owned by the same QA run;
replacement is atomic and never replaces separation or downstream source stages.
Export browser annotations before clearing browser storage or changing origin.

Manual categories are `better`, `same`, `worse`, `speaker_wrong`,
`dialogue_wrong`, `audio_wrong`, and `visual_hallucination`. Comparison labels
are mutually exclusive; issue labels can be combined.
Notes and labels persist in localStorage. **Export QA JSON** downloads
`h3_audio_shadow_qa.json`; **Import QA JSON** restores a matching export.
The sidecar shape is:

```json
{
  "schema_version": "r2v.h3.audio_shadow_human_qa.1",
  "dataset_fingerprint": "<sha256>",
  "shadow_run_id": "random10-v1",
  "annotations": [
    {
      "clip_uid": "<clip>",
      "record_fingerprint": "<sha256>",
      "labels": ["stem_issue"],
      "notes": "Manual observation"
    }
  ]
}
```

Exports follow case-manifest order and omit untouched clips. Fingerprints bind
the reviewed source facts; stale/unknown/duplicate imports fail rather than
attaching labels to changed artifacts. Annotations are independent QA sidecars,
never binding corrections or updates to DiariZen, ASR, facts, reconcile, or H3.

## Environment and overwrite rules

- Never install SAM-only, DiariZen, or Qwen3-ASR dependencies into the main
  `.venv` just to resolve a shadow-stage import error.
- Never replace the main Torch/CUDA/NumPy/Torchaudio stack as part of a pilot.
- Never write into `/mnt/workspace/public/dataset/**` or
  `/mnt/workspace/public/pretrained/**`.
- Do not hand-copy atomic stages between run IDs; stored absolute paths and
  hashes intentionally make such copies stale.
- Do not use `--overwrite` for a fresh named run. Before any intentional
  replacement, inspect the exact destination and all downstream lineage.
- Preserve the legacy/default one-clip smoke root as a regression baseline.

For semantic contracts, stage schemas, and authority rules, read
`docs/H3_SAM_AUDIO_STEM_SHADOW.md`. For production Qwen3-ASR behavior, read
`docs/H3_QWEN3_ASR.md`.
