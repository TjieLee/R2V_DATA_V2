# R2V_DATA_V2

A lightweight, sequential pipeline for constructing reference-conditioned video
training samples from existing clips.

The MVP follows one direct path:

```text
source JSON/JSONL
-> fixed ten-frame sampling
-> Qwen caption, entities, and explicit ref bindings
-> SAM3 text-prompted masks
-> hard gates, DINOv3 representativeness, optional SigLIP 2 alignment
-> Top-3 Qwen candidate review and code-owned final ranking
-> one canonical reference per retained entity
-> in-pair or verified same-parent cross-pair
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
- `qwen.base_url`: an OpenAI-compatible multimodal Qwen endpoint;
- `qwen.model`: the actual video-capable model served at that endpoint;
- `sam3.code_root`: the installed SAM3 checkout;
- `sam3.checkpoint`: an explicit local checkpoint path;
- `ranking.dinov3_model_path`: a verified local DINOv3 checkpoint or HF directory;
- `ranking.siglip2_model_path`: the explicitly downloaded local SigLIP 2 directory;
- `output_root`: a directory below `/mnt/workspace/litengjie/data/`.

The Qwen service launch command depends on the model and vLLM version installed
on the server. Do not use a language-model-only service for frame annotation.
The pipeline sends the same ten JPEG frames to Qwen and SAM3; it never sends a
Base64-encoded whole video.

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
  --stages manifest,frames,qwen,sam,rank,pair
```

Stages are ordinary Python functions called in order, not subprocesses. Existing
per-stage outputs are skipped; `--overwrite` rebuilds the selected stages. One
bad sample is written to the relevant JSONL log and does not stop its neighbors.

Each stage can also run directly:

```bash
python scripts/00_build_manifest.py --config configs/server.local.yaml --limit 20
python scripts/01_sample_frames.py --config configs/server.local.yaml
python scripts/02_qwen_annotate.py --config configs/server.local.yaml
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
├── references/<clip_uid>/<entity_id>/
├── samples/<clip_uid>.json
└── logs/
```

Annotation JSON files, reference `metadata.json` files, sample JSON files, and
augmentation sidecars are the durable stage artifacts. Their JSONL manifests
are rebuilt atomically at the end of each stage, so rerunning after an
interruption reconciles a completed artifact that was not yet indexed.

Every selected reference keeps:

- `canonical.jpg`: natural crop from the source frame;
- `mask.png`: selected crop mask;
- `foreground_rgba.png`: original foreground pixels with alpha;
- `neutral_background.jpg`: original foreground on light gray.

Candidate masks are stored as packed, zlib-compressed JSON for only the
shortlist. The final selected mask is the only mandatory PNG mask.
Stage 04 also stores per-candidate ranking metadata and float16 DINOv3
embeddings. The selected reference keeps `dinov3_embedding.npy` for downstream
reuse. DINOv3 and SigLIP 2 can each be disabled; their score weight is then
removed and the remaining weights are normalized. Q-Align is not part of this
pilot.

Cross-pair search is limited to the same `parent_video_id` and a different
complete numeric `clip_suffix`. It uses cached selected-reference DINOv3
embeddings for Top-10 coarse retrieval and falls back to color histograms when
either embedding is unavailable. Category/name evidence is checked before Qwen
dual-image exact-instance review. DINOv3 never decides identity: uncertain,
near-duplicate, conflicting, or low-confidence Qwen decisions fall back to
in-pair.

## Augmentation

Augmentation is disabled by default and never loads FLUX or Qwen-Image-Edit in
that mode. The module exposes small programmatic editor and validator callables
for a later server integration. Generated variants are accepted only when the
foreground core remains nearly unchanged and the supplied identity validator
passes after restoration; otherwise they are deleted while canonical references
remain intact. Each accepted sidecar records core similarity both before and
after original foreground pixels are restored. When DINOv3 is enabled, the
sidecar also records neutral-crop canonical-to-augmented cosine similarity as a
diagnostic; it is not a hard rejection threshold.

## Validate

No real Qwen or SAM3 model is needed for local tests:

```bash
python -m pytest -q
python -m ruff check .
```
