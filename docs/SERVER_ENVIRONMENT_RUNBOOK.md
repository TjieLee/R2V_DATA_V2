# V3 Server Environment Runbook

Last updated: 2026-08-28

This is the authoritative server handoff for the frozen Visual/reference line.
Verify the live remote branch and HEAD before operating the server; do not infer
server state from old chat history.

## Freeze identity

```text
repository: TjieLee/R2V_DATA_V2
Visual/reference branch: feature/v3-subject-attributes-v1
final Visual/reference code freeze: d056c32b76db4b3d7c0358b38e996e7a91a288d1

frozen original Visual branch: feature/v3-runtime-integrity-v1
frozen original Visual HEAD: 87bd4e06107d7f56df550979b0e96515cb70f911
core original Visual algorithm baseline: 3cfb11fdd1fbe4a5bbad02a775097d8ab3097288
```

Docs-only commits may advance branch HEAD without changing the code freeze.
Annotation production is frozen. Audio/H3 development remains on its own line.

## Confirmed paths

```text
repo:
  /mnt/workspace/litengjie/data/R2V_DATA_V2

python:
  /mnt/workspace/litengjie/data/R2V_DATA_V2/.venv/bin/python

SAM3 code:
  /mnt/workspace/litengjie/data/vendor/sam3
SAM3 checkpoint:
  /mnt/workspace/public/pretrained/facebook/sam3/sam3.pt

Qwen model:
  /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
Qwen endpoint:
  http://127.0.0.1:8000/v1

Boogu code:
  /mnt/workspace/litengjie/data/vendor/Boogu-Image
Boogu python:
  /mnt/workspace/litengjie/data/venvs/boogu-image/bin/python
Boogu model:
  /mnt/workspace/litengjie/data/models/Boogu-Image-0.1-Edit-Turbo-hotfix-1k-20260708
```

Production data:

```text
source JSONL:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/shots_f03_motion.jsonl
processed shot clips root:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808_processed/clips_clean_cropped
original/full source videos root:
  /mnt/workspace/public/dataset/jea-video/moive-183t-0808
```

`video_path` is the processed Visual input shot, currently MP4.
`source_video_path` is provenance for the original/full video, is
container-extension agnostic, and must never be substituted for `video_path` as
model input.

## Runtime allocation

```text
GPU 0-3: Qwen3-VL-32B-Instruct, BF16, TP1 x DP4
GPU 4: Boogu background removal
GPU 5 + 7: shared two-process SAM3 pool
GPU 6: Boogu reference_edit and eligible non-face Attribute completion
GME: disabled

Qwen max model length: 49152
runtime.qwen_max_inflight: 4
OMP_NUM_THREADS: 1
```

The SAM3 pool serves main temporal segmentation and Attribute single-frame
probes. Boogu loads persistently; it is not reloaded per request. Fresh
Subject/Object and Attribute generated-background calls are zero. Reference
completion and Attribute completion continue to use GPU 6.

## Shell environment

From the repository root:

```bash
export PYTHONPATH=/mnt/workspace/litengjie/data/vendor/sam3${PYTHONPATH:+:$PYTHONPATH}
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export OMP_NUM_THREADS=1
unset CUDA_VISIBLE_DEVICES
```

Do not add Boogu or GME to `PYTHONPATH`. Boogu uses its configured interpreter
and code root. GME is disabled.

## Safe repository update

Preserve all untracked server-local files. Do not run `git clean`, reset,
rebase, or force-push operations.

```bash
cd /mnt/workspace/litengjie/data/R2V_DATA_V2
git fetch origin
git switch feature/v3-subject-attributes-v1
git merge --ff-only origin/feature/v3-subject-attributes-v1
git status --short
git rev-parse HEAD
```

The branch HEAD may be a docs-only descendant of the code freeze. Inspect the
commit subjects before treating a newer HEAD as an algorithm change.

## Qwen service

Start or verify Qwen3-VL-32B-Instruct with:

```text
model: /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-32B-Instruct
dtype: bfloat16
tensor parallel size: 1
data parallel size: 4
visible GPUs: 0,1,2,3
max model length: 49152
endpoint: http://127.0.0.1:8000/v1
```

Confirm the endpoint before starting a fresh run. The production pipeline uses
`runtime.qwen_max_inflight: 4`.

## Fresh full run

The exact full stage order is:

```text
manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,subject_attributes,export
```

Set task-specific paths, then create only the config parent directory:

```bash
export CONFIG=/absolute/path/to/config.yaml
export RUN_ROOT=/absolute/path/to/new-run-root
export EXPORT_ROOT=/absolute/path/to/new-export-root
mkdir -p "$(dirname "$CONFIG")"
```

Do not precreate `RUN_ROOT`. In particular, do not precreate `EXPORT_ROOT`.
`DatasetExporter` requires the destination not to exist and publishes through a
temporary directory atomically.

Launch:

```bash
.venv/bin/python run_pipeline_v3.py \
  --config "$CONFIG" \
  --stages manifest,annotate,frames,segment,rank,background,remove,pair,reference_edit,reference_integrity,instruct,subject_attributes,export \
  --profile
```

If an existing run identity does not match the requested configuration or
model identity, use a new run root. Do not delete `run.json` and do not weaken
identity validation.

## Frozen selection policies

Subject/Object:

```text
complete or local_usable -> canonical alpha
repairable -> candidate 1 completion and comparative Qwen
  -> reject/not better: candidate 2 when available
  -> reject/not better: canonical alpha
selected completion integrity failure -> canonical alpha integrity
canonical alpha integrity failure -> bbox last resort
```

Completion never falls directly to bbox. Source-relative area, scale, and
center changes are diagnostic only for completion; identity, duplicate/wrong
instance, fragmentation, tiny/extreme placement, and severe warping remain
hard gates.

Attributes:

```text
at most 3 per eligible human owner
owner Top3 candidates; at most 2 different sources
single-frame SAM3 probes only; no temporal tracking
6 hard raw review flags, including sufficient_source_evidence
structure_complete and completion_recommended are diagnostics only
face: Boogu hard-disabled, accepted raw -> reviewed bbox -> raw alpha fallback
face without an accepted raw candidate -> reject; bbox cannot bypass raw gates
eligible non-face repair uses raw alpha + source RGB bbox + Boogu candidate
non-face insufficient evidence -> candidate 2, then bbox last resort
non-face bbox reject -> Attribute reject
fresh generated background -> disabled
```

See `V3_REFERENCE_EDIT_BOOGU.md` and `V3_SUBJECT_ATTRIBUTES_STATE.md` for exact
prompts and gates.

## Final validation evidence

Latest fixed-100 real-model run:

```text
run: e2e100-verify-324a29a-20260825-234558
commit used: 324a29aebcf4b573cab59332d337ec9d10ad9deb
```

```text
reference_edit background attempted/accepted: 0/0
reference_edit completion attempts: 5
candidate 1 accepted: 3
candidate 2 attempts: 0
fallback alpha: 2
reference_edit entities accepted/rejected: 95/0

reference_integrity entities accepted/rejected: 84/17
reference_integrity bbox fallback attempted/accepted: 1/1

attributes accepted: 84
attribute completion attempts/accepted: 70/51
attribute Qwen completion rejects: 19
attribute second candidate attempts: 21
attribute candidate 1/candidate 2 accepted: 46/5
attribute bbox fallback attempted/accepted: 3/0
attribute background variants: 0/0
```

The later validator fix, prompted by a one-owner/three-attribute batch-loss bug,
passed the full local suite but did not rerun this fixed-100 real-model canary.
The candidate-2 source-frame provenance fix at the earlier `d7f3d6b...` freeze
was validated locally with 368 focused and 1,991 full tests passing (one
warning), diff-check, and compileall. It also did not run a new GPU/model canary.

The current face bbox-first policy was validated locally with 148 focused and
2,000 full tests passing (one warning), compileall, and diff-check. Ruff found
the same 69 pre-existing issues on the clean parent and updated tree, so this
change introduced no new Ruff finding. No Qwen, SAM3, Boogu, GPU, or real-model
canary was run for it.

## Audio/H3 handoff

For cross-branch integration, consume compacted
`r2v.v3.production_sample.1`. Do not treat internal Attribute owner/review
sidecars as the final integration API. Read `V3_VISUAL_AUDIO_INTEGRATION.md`
before updating Audio/H3.
