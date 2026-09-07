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

The named run contains the same six owned stages:

```text
sam_audio_stem_shadow_v1/runs/<shadow-run-id>/
  separation/
  diarization/
  asr/
  mimo_stem_facts/
  mimo_reconcile/
  references/
```

All six commands must receive the same run ID and ordered case manifest. Hash
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

### 4. Three-stem MiMo facts

```bash
"$R2V_PYTHON" tools/run_h3_mimo25_stem_facts_shadow.py "${RUN_ARGS[@]}" \
  --model mimo-v2.5 \
  --base-url http://127.0.0.1:8092/v1 \
  --temperature 0.2 \
  --allow-unverified
```

### 5. Original-AV stem-aware reconcile

```bash
"$R2V_PYTHON" tools/run_h3_mimo25_stem_reconcile_shadow.py "${RUN_ARGS[@]}" \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --model mimo-v2.5 \
  --base-url http://127.0.0.1:8092/v1 \
  --max-completion-tokens 32768 \
  --allow-unverified
```

### 6. Stem-native primary voice references

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
cat "$SHADOW_ROOT/mimo_stem_facts/summary.json"
cat "$SHADOW_ROOT/mimo_reconcile/summary.json"
cat "$SHADOW_ROOT/references/references.jsonl"
```

Expected stage ownership is strictly run-local. If an upstream stage is
intentionally replaced, all already-published downstream stages become stale by
lineage and must be inspected before any explicit replacement. Prefer a new run
ID for a new case inventory or algorithm experiment.

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
