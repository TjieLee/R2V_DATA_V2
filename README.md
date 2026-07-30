# R2V_DATA_V2

A lightweight, sequential pipeline for constructing reference-conditioned video
training samples from existing clips.

The MVP follows one direct path:

```text
source JSON/JSONL
-> Qwen full-video caption, entities, and explicit ref bindings
-> independent fixed ten-frame sampling
-> SAM3 text-prompted masks
-> hard gates, DINOv3 representativeness, optional SigLIP 2 alignment
-> category-aware Qwen candidate review and code-owned final ranking
-> entity and optional background canonical references
-> optional local FLUX.1 Fill repair with strict pixel preservation
-> entity in-pair/cross-pair plus background in-pair binding
-> final_samples.jsonl
```

It intentionally has no runtime factory, plugin system, Gold Judge, watermark
workflow, evidence chain, state machine, or complex resume manager.

## Install

Python 3.12 or newer is required.

```bash
git clone https://github.com/TjieLee/R2V_DATA_V2.git
cd R2V_DATA_V2
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Install the optional local vision-model clients without changing the server's
Torch or CUDA packages:

```bash
.venv/bin/pip install -r requirements-vision.txt
```

FLUX.1 Fill is a separate opt-in install. Use it only in the server environment
that already has the intended Torch build:

```bash
.venv/bin/pip install -r requirements-inpaint.txt
```

The current upstream SAM3 package has its own CUDA, PyTorch, and Python
requirements. Install the repository already present at the configured
`sam3.code_root` into the server environment; do not let this project replace
the server's global Torch/CUDA packages.

`requirements.txt` is deliberately lightweight. It does not install or upgrade
Torch, torchvision, torchaudio, CUDA, cupy, nixl, flash-attn, or SAM3. Manage
those packages in the existing GPU server environment.

## Configure

Copy the example and edit only machine-specific values:

```bash
cp configs/default.yaml configs/server.local.yaml
```

Required server inputs:

- `dataset_json`: source JSON or JSONL;
- `qwen.annotation`: the OpenAI-compatible video annotation endpoint/model;
- `qwen.candidate_judge`, `qwen.background_judge`, and
  `qwen.cross_pair_judge`: independently replaceable image endpoints/models;
- `sam3.code_root`: the installed SAM3 checkout;
- `sam3.checkpoint`: an explicit local checkpoint path;
- `ranking.dinov3_model_path`: required when DINOv3 is enabled; use a verified
  local checkpoint or complete HF directory;
- `ranking.siglip2_model_path`: required when SigLIP 2 is enabled; use the
  explicitly downloaded local directory;
- `output_root`: a directory below `/mnt/workspace/litengjie/data/`.

The Qwen service launch command depends on the model and vLLM version installed
on the server. Do not use a language-model-only service for video annotation.
Qwen receives a local `file://` URI for the complete source video while SAM3 and
reference ranking use the independently sampled ten JPEG frames. The client
does not Base64-encode the video or depend on the frame directory.

The trusted server must explicitly allow vLLM to read the read-only dataset
root, for example:

```bash
vllm serve /mnt/workspace/public/pretrained/Qwen/Qwen3-VL-8B-Instruct \
  --allowed-local-media-path /mnt/workspace/public/dataset
```

Configure Qwen's internal video processing independently from SAM3 sampling:

```yaml
qwen:
  annotation:
    base_url: http://127.0.0.1:8000/v1
    model: /mnt/workspace/public/pretrained/Qwen/<video-model>
    video:
      input_mode: full_video
      fps: 2.0
      do_sample_frames: true
      max_pixels: null
      total_pixels: null
  candidate_judge:
    base_url: http://127.0.0.1:8001/v1
    model: /mnt/workspace/public/pretrained/Qwen/<image-model>
```

The `fps`, optional pixel budgets, and internal sampling are forwarded to the
Qwen/vLLM media processor. They do not change `frames.count`, which remains ten.
The former flat `qwen` mapping remains accepted and is mapped to every service
for compatibility.

Keep model and package caches in the writable user directory, for example:

```bash
export HF_HOME=/mnt/workspace/litengjie/data/cache/huggingface
export TORCH_HOME=/mnt/workspace/litengjie/data/cache/torch
export XDG_CACHE_HOME=/mnt/workspace/litengjie/data/cache/xdg
export TMPDIR=/mnt/workspace/litengjie/data/tmp
```

The following roots are inputs and must remain read-only:

```text
/mnt/workspace/public/dataset/
/mnt/workspace/public/pretrained/
```

Inspect the available DINOv3 checkpoints before setting
`ranking.dinov3_model_path`; the pipeline never guesses or downloads one:

```bash
find /mnt/workspace/public/pretrained/dinov3 \
  -maxdepth 4 -type f \
  \( -name "*.pth" -o -name "*.pt" -o -name "*.safetensors" -o -name "config.json" \) \
  | sort | head -100
```

The repository default disables both optional visual models. A server pilot
must enable them explicitly with real local paths:

```yaml
ranking:
  evaluators:
    qwen_visual:
      enabled: true
      use_for_final_score: true
    dinov3:
      enabled: true
      use_for_preselection: true
      use_for_final_score: true
      hard_reject_outlier: false
    siglip2:
      enabled: true
      use_for_preselection: true
      use_for_final_score: true
      hard_reject_wrong_entity: false

  dinov3_repo_dir: /mnt/workspace/public/pretrained/dinov3
  dinov3_model_path: /mnt/workspace/public/pretrained/dinov3/<actual_checkpoint>
  dinov3_model_name: dinov3_vits16

  siglip2_model_path: /mnt/workspace/litengjie/data/models/siglip2-base-patch16-naflex
```

Evaluator execution, preselection participation, final-score participation, and
pilot hard gates are separate switches. `ranking.preselection_weights` and
`ranking.final_weights` accept only the metric names shown in
`configs/default.yaml`; enabled positive weights are normalized at runtime, so
they do not need to sum to one. Metric scaling is explicit under
`ranking.normalization` and supports only `identity`, `minmax`,
`robust_minmax`, and `fixed_range`. Hard gates always use raw measurements or
raw model judgments, never candidate-relative normalized values.

SigLIP 2 is also local-only at runtime. Download it explicitly into the
writable user model directory:

```bash
python scripts/download_optional_models.py \
  --siglip2 google/siglip2-base-patch16-naflex \
  --destination /mnt/workspace/litengjie/data/models/siglip2-base-patch16-naflex
```

The downloader rejects destinations outside the user model and Hugging Face
cache roots. It never writes to `/mnt/workspace/public/pretrained/`.

## Run

Run the first 20 records:

```bash
python run_pipeline.py \
  --config configs/server.local.yaml \
  --limit 20 \
  --stages manifest,qwen,frames,sam,rank,background,pair
```

Stages are ordinary Python functions called in order, not subprocesses. Existing
per-stage outputs are skipped; `--overwrite` rebuilds the selected stages. One
bad sample is written to the relevant JSONL log and does not stop its neighbors.

Each stage can also run directly:

```bash
python scripts/00_build_manifest.py --config configs/server.local.yaml --limit 20
python scripts/02_qwen_annotate.py --config configs/server.local.yaml
python scripts/01_sample_frames.py --config configs/server.local.yaml
python scripts/03_sam3_extract.py \
  --config configs/server.local.yaml \
  --sam3-checkpoint /path/to/explicit/checkpoint.pt
python scripts/04_rank_references.py --config configs/server.local.yaml
python scripts/05_build_pairs.py --config configs/server.local.yaml
python scripts/06_augment_references.py --config configs/server.local.yaml
```

## Outputs

```text
<output_root>/
├── manifests/
│   ├── source.jsonl
│   ├── annotations.jsonl
│   ├── references.jsonl
│   └── final_samples.jsonl
├── frames/<clip_uid>/
├── annotations/<clip_uid>.json
├── candidates/<clip_uid>/<entity_id>/
├── background_candidates/<clip_uid>/
├── references/<clip_uid>/<entity_id-or-bg1>/
├── samples/<clip_uid>.json
└── logs/
```

Annotation JSON files, reference `metadata.json` files, sample JSON files, and
augmentation sidecars are the durable stage artifacts. Their JSONL manifests
are rebuilt atomically at the end of each stage, so rerunning after an
interruption reconciles a completed artifact that was not yet indexed.

Every selected reference keeps:

- `canonical_raw.jpg`: immutable pre-repair source reference;
- `canonical.jpg`: natural crop from the source frame;
- `mask.png`: selected crop mask;
- `foreground_rgba.png`: original foreground pixels with alpha;
- `neutral_background.jpg`: original foreground on light gray.

SAM tracking masks from all sampled slots are stored separately from masks that
pass entity-reference size and area gates. Background removal consumes the
tracking masks, while entity ranking consumes only the candidate masks.
Per-slot coverage and filtering reasons are recorded in `mask_coverage.json`.
Both mask sets use packed, zlib-compressed JSON. `candidate_status.json` is the
publication gate for ranking; visibility below `sam3.minimum_visible_frames`
keeps tracking artifacts for background use but publishes empty entity
candidate artifacts. SAM overwrite clears the entire entity candidate
directory before tracking.

Legacy background fallback from `top_masks.rle.json` is disabled by default.
It must be explicitly enabled with `background.allow_legacy_candidate_masks`,
and missing legacy slots are still marked incomplete. Qwen-reviewed candidates
with incomplete masks are hard rejected above the configured strict foreground
distraction threshold. The final selected mask is the only mandatory PNG mask.

Before entity inpainting, `mask.png`, `foreground_rgba.png`,
`neutral_background.jpg`, and `dinov3_embedding.npy` are snapshotted to
corresponding `_raw` artifacts. Overwrite and fallback restore these immutable
copies. Inpainting metadata binds each result to the source image, source mask,
source frame index, reference semantics, effective prompts, metadata version,
and inpainting configuration.
Stage 04 also stores per-candidate ranking metadata and float16 DINOv3
embeddings. The selected reference keeps `dinov3_embedding.npy` for downstream
reuse. DINOv3 and SigLIP 2 can each be disabled; their score weight is then
removed and the remaining weights are normalized. Q-Align is not part of this
pilot. Qwen completeness, recognizability, mask quality, visual quality, and
inverse occlusion all contribute to the code-owned final score.
Canonical view is a preferred tier among otherwise valid candidates; it is not
a front-view hard gate. Border contact is a soft completeness score by default
and can still be restored as a hard gate with `ranking.reject_border_touch`.
`qwen_suggested_best_frame_slot` is retained only as a diagnostic and never
overrides hard gates or the final weighted ordering.

`ranking_metadata.json` records each candidate's `raw_scores`,
`normalized_scores`, `preselection_score`, and `final_score`. Its top level
records the effective preselection and final weights plus all normalization
policies. The selected reference metadata repeats its raw and normalized score
composition and `effective_final_weights`, so pilot decisions remain
explainable after manifests are reconciled.

Cross-pair search is limited to the same `parent_video_id` and a different
complete numeric `clip_suffix`. It uses cached selected-reference DINOv3
embeddings for Top-10 coarse retrieval and falls back to color histograms when
either embedding is unavailable. Category/name evidence is checked before Qwen
dual-image exact-instance review. DINOv3 never decides identity: uncertain,
near-duplicate, conflicting, or low-confidence Qwen decisions fall back to
in-pair.

Background references use `<ref_bg_1>`, stay in-pair, and are added only when
their phrase has one exact caption binding. An unbindable background is dropped
with a warning without dropping valid entity references. The optional
`inpaint` stage loads FLUX only when `inpainting.enabled: true` and an explicit
existing local `model_path` is configured. Generated pixels are accepted only
inside the repair mask; output outside that mask remains exactly equal to the
raw reference. References have an explicit `ready`, `pending_inpainting`, or
`rejected` status, and only `ready` records may enter pairing or augmentation.
Final samples require at least one entity reference by default
(`pairing.require_entity_reference: true`). Background-only artifacts remain in
`references.jsonl` and under `references/<clip_uid>/bg1/`, but are excluded from
`samples/*.json` and `final_samples.jsonl`.
Background masking includes every separable visible entity, and a clean raw
background is preferred over a higher-scoring candidate that requires repair.
Backgrounds that require repair remain pending until FLUX succeeds and semantic
consistency checks pass; a failed background repair is rejected instead of
falling back to the contaminated raw frame. Production FLUX background hole
filling requires an explicit video-capable `qwen.repair_judge`; DINOv3 and
SigLIP2 alone remain sufficient only for entity-only validation when background
repair is disabled. Successful entity repair rebuilds its mask, RGBA, neutral
background, and selected DINO embedding so downstream artifacts remain aligned.
Background hole-fill review verifies foreground removal and coherent scene
continuation instead of entity identity preservation. Its source foreground
area is gated before any transformation
(`inpainting.background.maximum_hole_area_ratio: 0.23`). FLUX receives a
separate hole-filled, closed, component-grouped, per-group convex-hull mask
with adaptive dilation, capped by
`inpainting.background.maximum_generation_mask_area_ratio: 0.35`. Both source
and generation mask paths and ratios remain in metadata. The full-frame Qwen
scene review and the independent foreground-removal and continuity reviews for
each context-upscaled generation-mask component must all pass. Foreground
removal compares original/repaired mask-only crops, a repaired context crop,
and the binary mask. Comparison sheets, overlays, and difference heatmaps are
retained only for human inspection and are never sent to Qwen. Object-like
annotation entities also receive a repaired-image SAM3 grounding check; at
least 20 percent overlap with the generation mask rejects the candidate as
remaining or reconstructed foreground.
Background FLUX prompts are generated from a context-only image whose generation
mask pixels are neutral gray, are validated against foreground and
negative-language terms, and never include the caption or reference phrase.
Qwen prompt generation fails closed; generic prompting is used only when
`prompt_mode: generic` is explicitly configured. Configured seeds are evaluated
independently; every candidate and its inner/outer boundary diagnostics are
retained under `inpainting_candidates/`. When all seeds are evaluated, accepted
candidates are shown in a contact sheet and the earliest accepted seed is used
only as a compatibility publication choice, explicitly marked
`first_accepted_not_quality_ranked`; background DINO scores never rank them.
`canonical_repaired_candidate.png` remains the compatibility inspection path,
including after rejection.

## Qwen Benchmark

Compare already-served OpenAI-compatible video backends on a fixed source
manifest without starting vLLM:

```bash
python scripts/benchmark_qwen_backends.py \
  --source-manifest /path/to/source.jsonl \
  --clip-uids-file /path/to/fixed_clip_uids.txt \
  --backend large http://127.0.0.1:8000/v1 served-large-model \
  --backend small http://127.0.0.1:8002/v1 served-small-model \
  --output-dir benchmarks/qwen
```

Each invocation creates a unique JSON summary and per-clip JSONL record. Prior
runs are never overwritten. The script only calls endpoints and models supplied
on the command line; it does not launch services or download weights.

FLUX background-fill sweeps use a separate output root and refuse any path
inside the production pipeline output:

```bash
python scripts/benchmark_flux_background_fill.py \
  --config configs/default.yaml \
  --output-dir /path/to/benchmarks/flux-background \
  --prompt-modes empty generic qwen_local \
  --guidance-scales 20 30 \
  --steps 50 \
  --seeds 0 17
```

The benchmark writes per-candidate images, masks, comparisons, JSONL, CSV, and
a summary without changing production references, manifests, or samples. It
reuses one loaded FLUX pipeline across every guidance/step combination.
`prompt_mode=empty` sends an actual empty string, while Qwen prompt failures
reject that benchmark candidate instead of substituting a generic prompt.

Existing benchmark candidates can be re-reviewed with the current validator
without running FLUX or modifying the candidate images:

```bash
python scripts/revalidate_flux_background_candidates.py \
  --config configs/default.yaml \
  --run-dir /path/to/benchmarks/flux-background/flux_background_fill_RUN
```

Each invocation creates a new versioned revalidation directory beside the
candidate manifest and records `flux_inference_performed: false` in its summary.

## Augmentation

Augmentation is disabled by default and never loads FLUX or Qwen-Image-Edit in
that mode. The module exposes small programmatic editor and validator callables
for a later server integration. Generated-background variants restore the
canonical foreground core before edge and identity validation and require at
least 0.995 post-restore core similarity. Viewpoint variants keep the generated
subject unchanged, then run the validator and optional DINOv3 diagnostic on that
generated output. Rejected variants are deleted while canonical references
remain intact. Each accepted sidecar records pre- and final-core similarity plus
`foreground_core_restored`. When DINOv3 is enabled, the sidecar also records
neutral-crop canonical-to-augmented cosine similarity as a diagnostic; it is not
a hard rejection threshold.

## Validate

No real Qwen or SAM3 model is needed for local tests:

```bash
python -m pytest -q
python -m ruff check .
```
