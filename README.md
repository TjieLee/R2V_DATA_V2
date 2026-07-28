# R2V_DATA_V2

Lightweight reference-to-video data construction pipeline. The implementation
is intentionally a sequence of small Python stages with JSON/JSONL artifacts.

The first stage builds a normalized source manifest without modifying the input
dataset:

```bash
python scripts/00_build_manifest.py \
  --config configs/default.yaml \
  --limit 20
```

The public dataset and pretrained-model directories are read-only. All outputs
must be configured below `/mnt/workspace/litengjie/data/`.

