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
  mimo_reconcile_stemtext_final_av_v34/
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

### 4. Two audio-only candidates, then one final original AV

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

Two audio-only requests first run concurrently (max two threads). Each receives
one canonical SAM audio URL and the v33 positive-only role-blind prompt, no video,
image, ASR, references, caption or ICL. Hashes are checked against separation
provenance. Temperature 0, disabled thinking, 1024 tokens and one-string JSON
remain unchanged. Results/errors are stored by music/SFX pipeline role.

Only after both requests finish or fail, one final AV request receives the
original target video with embedded audio, existing reference images, unchanged
authoritative text facts and official opening + Shot 1 ICL, plus:
`music_separator_candidate` and `sfx_separator_candidate`. Missing evidence is
`SOURCE_UNAVAILABLE`, never silence. Generated candidates are ephemeral request
text, not fields added to `MimoClipJob`. No independent audio enters final AV.

Original AV is a factual contradiction filter, not a requirement to independently
re-prove every auxiliary detail. Preserve positive acoustic observations by
default, including weak/masked sounds and harmless perceptual wording. Correct
concrete factual errors or clear leakage/artifacts; an unsupported engine guess
may become a low rumble without deleting the sound. Auxiliary absence never
establishes absence. Score goes to `non_diegetic_music`; in-scene music is
authored naturally in `shot1_caption`, never soundscape. Uncertain music placement
uses conservative caption prose, not guessed BGM. Soundscape remains non-musical,
non-dialogue ambience/SFX, with no programmatic semantic checker.

Final AV directly produces `overall_soundscape` and `non_diegetic_music`
(`N/A` for no eligible content). There is no intermediate `sound_description`
or text fusion. The materializer reads these annotation fields directly.
Caption, grounding, relaxed dialogue, multi-speaker exclusion and six-section
Ref2VA formatting remain unchanged.

Normal per-clip counts: audio=2, AV=1, total=3. One attempt per request, no
SDK retry, repair, fallback, AV recheck or resend. Auxiliary errors do not
automatically fail a clip; final AV still runs. Final AV failures retain raw
annotation/error and both candidates, without a fourth call. Zero/missing
embedded audio tokens remain warnings only. AV defaults still include
`--thinking disabled --icl official_ref2va_v1`, `use_audio_in_video=true`,
temperature 0.0 and 32768 completion tokens.

Versions: prompt v34, annotation .20, backend .36, materializer v23, authority
v17, ICL v2; reconcile record .10, summary .12, policy v5. New output:
`mimo_reconcile_stemtext_final_av_v34/`. Old
`mimo_reconcile_av_stemtext_sound_partition/`, `mimo_v29_oneclip_smoke/`,
and `mimo_v30_833_oneclip_smoke/` are preserved, not migrated.

For a fresh v34 random10 run, keep `CASE_MANIFEST` as the existing ordered
random10 manifest. This reuses SAM/DiariZen/ASR without reruns and leaves the
previous `mimo_reconcile_stemtext_final_av_v33/` and earlier outputs untouched:

```bash
"$R2V_PYTHON" tools/run_h3_mimo25_stem_reconcile_shadow.py \
  --visual-production-root "$VISUAL_PRODUCTION_ROOT" \
  --visual-runs-root "$VISUAL_RUNS_ROOT" \
  --audio-production-root "$AUDIO_PRODUCTION_ROOT" \
  --shadow-run-id random10-v1 --case-manifest "$CASE_MANIFEST" \
  --sam-route music_first --allow-unverified \
  --model mimo-v2.5 --base-url http://127.0.0.1:8092/v1 \
  --media-root /mnt/workspace --max-completion-tokens 32768 \
  --output-root "$AUDIO_PRODUCTION_ROOT/sam_audio_stem_shadow_v1/runs/random10-v1/mimo_reconcile_stemtext_final_av_v34"
```

No `--overwrite` is used. This patch has fake-client coverage only; readiness
and quality require a fresh server run and manual QA.

V33 sends the validator's exact, stably sorted `allowed_h3_reference_labels`
to final AV; absent Audio labels are forbidden. Sx denotes actual vocal sources,
not Subject indexes: silent Subjects consume no speaker ID.

V34 allows a missing repeated marker only when dialogue-block count equals the
chronological speech-fact count and the adjacent speaker IDs are identical.
The first block and speaker transitions still require an explicit known Sx.
Unequal counts do not fail by themselves: already marked merged dialogue stays
valid, but unmarked blocks receive no inferred mapping. No pronoun parsing,
transcript matching, text rewrite or model call is added.

Only repeated Subject labels in definition/retention prose and internal
segment/visual inventory mismatches with zero transcribed segments are relaxed.
The latter are retained as diagnostic warnings; speech inventory, unknown
references/speakers, dialogue formatting, articulation contradictions and unsafe
multi-speaker identity publication remain hard gates. A failed AV record keeps
the backend's already parsed/canonicalized annotation for QA, without becoming
ready or causing another call.

Both role-blind audio prompts now request only positive acoustic observations,
not absence summaries or guessed causes, locations or recording history.
Final AV must not copy track-local absence claims, must remove unsupported
source interpretations while preserving acoustic content, and must never put
music in overall_soundscape. These are prompt instructions, not code checkers.

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
cat "$SHADOW_ROOT/mimo_reconcile_stemtext_final_av_v34/summary.json"
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
export QA_ROOT="$AUDIO_PRODUCTION_ROOT/../h3-audio-qa/$SHADOW_RUN_ID-stemtext-final-av"
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
Final raw caption and AV sound fields remain visible for parseable failed records;
music/SFX candidates and errors, final AV sound fields/raw/error, and
AV/audio call counts are shown separately.

The exact final six-section prompt uses the existing six-section materializer v23 path
when AV validation permits it. Each source conditioning variant
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
